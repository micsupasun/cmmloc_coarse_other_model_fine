"""Language encoder from the user's CMMLoc_MNCLv4 fine-stage source.

This intentionally lives outside ``models.fine`` so CMMLoc and the custom
model cannot silently share forward implementations just because their
checkpoint parameter names happen to match.
"""

from contextlib import nullcontext
from typing import List

import torch
import torch.nn as nn
from nltk import tokenize as text_tokenize
from transformers import AutoTokenizer, T5EncoderModel


def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct an MLP with a trailing ReLU."""
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[index - 1], channels[index]),
                    nn.BatchNorm1d(channels[index]),
                    nn.ReLU(),
                )
                for index in range(1, len(channels))
            ]
        )
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[index - 1], channels[index]),
                nn.ReLU(),
            )
            for index in range(1, len(channels))
        ]
    )


def get_mlp2(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct an MLP without a ReLU in the final layer."""
    layers = []
    for index in range(1, len(channels)):
        is_final = index == len(channels) - 1
        parts = [nn.Linear(channels[index - 1], channels[index])]
        if add_batchnorm:
            parts.append(nn.BatchNorm1d(channels[index]))
        if not is_final:
            parts.append(nn.ReLU())
        layers.append(nn.Sequential(*parts))
    return nn.Sequential(*layers)


class LanguageEncoder(torch.nn.Module):
    """T5 encoder and projection used when the custom fine model was trained."""

    def __init__(
        self,
        embedding_dim,
        hungging_model=None,
        fixed_embedding=False,
        intra_module_num_layers=2,
        intra_module_num_heads=4,
        is_fine=False,
        inter_module_num_layers=2,
        inter_module_num_heads=4,
        text_max_length=128,
        prealign_mlp_path=None,
    ):
        super().__init__()
        self.is_fine = is_fine
        self.model_name = hungging_model or "t5-large"
        self.text_max_length = text_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = text_max_length
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.llm_model = T5EncoderModel.from_pretrained(self.model_name)
        self.fixed_embedding = fixed_embedding
        if fixed_embedding:
            for parameter in self.llm_model.parameters():
                parameter.requires_grad = False

        input_dim = self.llm_model.encoder.embed_tokens.weight.shape[-1]
        self.intra_module = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    input_dim,
                    intra_module_num_heads,
                    dim_feedforward=input_dim * 4,
                )
                for _ in range(intra_module_num_layers)
            ]
        )
        self.inter_mlp = get_mlp2(
            [input_dim, embedding_dim],
            add_batchnorm=True,
        )
        if prealign_mlp_path:
            projection_state = torch.load(
                prealign_mlp_path,
                map_location="cpu",
            )
            projection_state = {
                key.split("inter_mlp.")[1]: value
                for key, value in projection_state.items()
                if "batches" not in key
            }
            self.inter_mlp.load_state_dict(projection_state)

        if not is_fine:
            self.inter_module = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        embedding_dim,
                        inter_module_num_heads,
                        dim_feedforward=embedding_dim * 4,
                    )
                    for _ in range(inter_module_num_layers)
                ]
            )

    def forward(self, descriptions):
        sentences = []
        for description in descriptions:
            sentences.extend(text_tokenize.sent_tokenize(description))

        batch_size = len(descriptions)
        num_sentences = len(sentences) // batch_size
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.text_max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        llm_context = torch.no_grad if self.fixed_embedding else nullcontext
        with llm_context():
            with torch.cuda.amp.autocast(enabled=False):
                output = self.llm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                )
                encodings = output.last_hidden_state.float()

        if self.fixed_embedding:
            encodings = encodings.detach()
        encodings = torch.nan_to_num(
            encodings,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        encodings = encodings.permute(1, 0, 2)
        for layer in self.intra_module:
            encodings = layer(encodings)
            encodings = torch.nan_to_num(
                encodings,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        encodings = encodings.permute(1, 0, 2).contiguous()
        encodings = encodings.max(dim=1)[0]
        encodings = self.inter_mlp(encodings)
        encodings = torch.nan_to_num(
            encodings,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        encodings = encodings.view(batch_size, num_sentences, -1)
        if self.is_fine:
            return encodings

        encodings = encodings.permute(1, 0, 2)
        for layer in self.inter_module:
            encodings += layer(encodings)
            encodings = torch.nan_to_num(
                encodings,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        return encodings.max(dim=0)[0]

    @property
    def device(self):
        return next(self.inter_mlp.parameters()).device

