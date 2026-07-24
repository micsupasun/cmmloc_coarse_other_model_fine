"""Custom fine matcher reproduced from CMMLoc_MNCLv4."""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The v4 transformer file differs from the CMMLoc copy only in an unrelated
# ``Transformer`` helper. The ``TransformerDecoderLayer`` used here is
# byte-for-byte equivalent, so reuse it without duplicating CUDA-only code.
from models.fine.transformer import TransformerDecoderLayer
from models.my_model.language_encoder import LanguageEncoder, get_mlp
from models.my_model.object_encoder import ObjectEncoder


def get_mlp_offset(
    dimensions: List[int],
    add_batchnorm=False,
) -> nn.Sequential:
    """Return an MLP with no activation after the offset output."""
    if len(dimensions) < 3:
        print("get_mlp_offset(): less than 2 layers!")
    layers = []
    for index in range(len(dimensions) - 1):
        layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
            if add_batchnorm:
                layers.append(nn.BatchNorm1d(dimensions[index + 1]))
    return nn.Sequential(*layers)


def get_direction(position_a, position_b):
    difference = position_a - position_b
    if np.linalg.norm(difference[0:2]) < 0.05:
        direction = "on-top"
    else:
        # Preserve v4's ordering exactly, including the diagonal tie behavior.
        if abs(difference[0]) >= abs(difference[1]) and difference[0] >= 0:
            direction = "east"
        if abs(difference[0]) >= abs(difference[1]) and difference[0] <= 0:
            direction = "west"
        if abs(difference[0]) <= abs(difference[1]) and difference[1] >= 0:
            direction = "north"
        if abs(difference[0]) <= abs(difference[1]) and difference[1] <= 0:
            direction = "south"
    return direction


class CrossMatch(torch.nn.Module):
    """CMMLoc_MNCLv4 fine architecture and forward preprocessing."""

    def __init__(
        self,
        known_classes: List[str],
        known_colors: List[str],
        args,
    ):
        super().__init__()
        self.embed_dim = args.fine_embed_dim
        self.object_encoder = ObjectEncoder(
            args.fine_embed_dim,
            known_classes,
            known_colors,
            args,
        )
        self.language_encoder = LanguageEncoder(
            args.fine_embed_dim,
            hungging_model=args.hungging_model,
            fixed_embedding=args.fixed_embedding,
            intra_module_num_layers=args.fine_intra_module_num_layers,
            intra_module_num_heads=args.fine_intra_module_num_heads,
            is_fine=True,
            text_max_length=args.text_max_length,
            prealign_mlp_path=args.prealign_mlp_path,
        )
        self.mlp_offsets = get_mlp_offset(
            [self.embed_dim, self.embed_dim // 2, 2]
        )
        self.loc_transform = get_mlp(
            [self.embed_dim, self.embed_dim // 2, 1]
        )
        self.direction_vocab = [
            "east",
            "west",
            "north",
            "south",
            "on-top",
        ]
        self.direction_to_idx = {
            name: index
            for index, name in enumerate(self.direction_vocab)
        }
        self._cached_direction_embeddings = None

        if args.fine_num_decoder_layers > 0:
            self.cross_hints = nn.ModuleList(
                [
                    nn.TransformerDecoderLayer(
                        d_model=args.fine_embed_dim,
                        nhead=args.fine_num_decoder_heads,
                        dim_feedforward=args.fine_embed_dim * 4,
                    )
                    for _ in range(args.fine_num_decoder_layers)
                ]
            )
            self.cross_objects = nn.ModuleList(
                [
                    TransformerDecoderLayer(
                        d_model=args.fine_embed_dim,
                        nhead=args.fine_num_decoder_heads,
                        dim_feedforward=args.fine_embed_dim * 4,
                    )
                    for _ in range(args.fine_num_decoder_layers)
                ]
            )
        else:
            self.cross_hints = nn.TransformerDecoderLayer(
                d_model=args.fine_embed_dim,
                nhead=args.fine_num_decoder_heads,
                dim_feedforward=args.fine_embed_dim * 4,
            )
            self.cross_objects = None

    def _get_direction_embeddings(self):
        if (
            self.language_encoder.fixed_embedding
            and self._cached_direction_embeddings is not None
            and self._cached_direction_embeddings.device == self.device
        ):
            return self._cached_direction_embeddings
        embeddings = self.language_encoder(
            self.direction_vocab
        ).squeeze(1)
        if self.language_encoder.fixed_embedding:
            self._cached_direction_embeddings = embeddings.detach()
        return embeddings

    def forward(self, objects, hints, object_points):
        batch_size = len(objects)
        num_objects = len(objects[0])
        hint_encodings = self.language_encoder(hints)
        (
            object_encodings,
            position_embeddings,
            relative_positions,
        ) = self.object_encoder(objects, object_points)

        relative_positions = relative_positions.to(object_encodings.device)
        object_embeddings = object_encodings.reshape(
            batch_size,
            num_objects,
            self.embed_dim,
        )
        object_embeddings = F.normalize(object_embeddings, dim=-1)
        position_embeddings = position_embeddings.reshape(
            batch_size,
            num_objects,
            self.embed_dim,
        )
        position_embeddings = F.normalize(
            position_embeddings,
            dim=-1,
        ).transpose(0, 1)

        directions = []
        for objects_sample in objects:
            for object_a in objects_sample:
                for object_b in objects_sample:
                    directions.append(
                        get_direction(
                            object_a.get_center(),
                            object_b.get_center(),
                        )
                    )
        direction_bank = self._get_direction_embeddings()
        direction_indices = torch.tensor(
            [
                self.direction_to_idx[direction]
                for direction in directions
            ],
            dtype=torch.long,
            device=direction_bank.device,
        )
        direction_embeddings = direction_bank.index_select(
            0,
            direction_indices,
        )
        location_information = self.loc_transform(
            direction_embeddings
        ).squeeze().reshape(
            batch_size,
            num_objects,
            num_objects,
        )
        location_information = F.normalize(
            location_information,
            dim=-1,
        ).to(object_encodings.device)
        relative_positions = relative_positions + 0.1 * location_information
        relative_positions = F.normalize(
            relative_positions,
            dim=-1,
        ).repeat(4, 1, 1).to(object_encodings.device)

        object_descriptors = object_embeddings.transpose(0, 1)
        hint_descriptors = hint_encodings.transpose(0, 1)
        if self.cross_objects is not None:
            for object_layer, hint_layer in zip(
                self.cross_objects,
                self.cross_hints,
            ):
                object_descriptors = object_layer(
                    object_descriptors,
                    hint_descriptors,
                    relative_position=relative_positions,
                    query_pos=position_embeddings,
                )
                hint_descriptors = hint_layer(
                    hint_descriptors,
                    object_descriptors,
                )
        else:
            hint_descriptors = self.cross_hints(
                hint_descriptors,
                object_descriptors,
                position_embeddings,
            )

        return self.mlp_offsets(hint_descriptors.max(dim=0)[0])

    @property
    def device(self):
        return next(self.mlp_offsets.parameters()).device
