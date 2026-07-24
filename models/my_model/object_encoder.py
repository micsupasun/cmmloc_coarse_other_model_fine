"""Object encoder from the user's CMMLoc_MNCLv4 fine-stage source."""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from datapreparation.kitti360pose.imports import Object3d
from datapreparation.kitti360pose.utils import COLOR_NAMES
from models.my_model.language_encoder import get_mlp
from models.pointcloud.pointnet2 import PointNet2


def safe_l2_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Match v4's finite float32 normalization."""
    tensor = torch.nan_to_num(
        tensor.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return F.normalize(tensor, p=2, dim=dim, eps=1e-6)


class ObjectEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        known_classes: List[str],
        known_colors: List[str],
        args,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.args = args

        self.known_classes = {
            class_name: index + 1
            for index, class_name in enumerate(known_classes)
        }
        self.known_classes["<unk>"] = 0
        self.class_embedding = torch.nn.Embedding(
            len(self.known_classes),
            embed_dim,
            padding_idx=0,
        )
        self.known_colors = {
            color_name: index
            for index, color_name in enumerate(COLOR_NAMES)
        }
        self.known_colors["<unk>"] = 0
        self.color_embedding = torch.nn.Embedding(
            len(self.known_colors),
            embed_dim,
            padding_idx=0,
        )

        self.pos_encoder = get_mlp([3, 64, embed_dim])
        self.color_encoder = get_mlp([3, 64, embed_dim])
        if args.prealign_color_path:
            color_state = torch.load(
                args.prealign_color_path,
                map_location="cpu",
            )
            color_state = {
                key.split("color_encoder.")[1]: value
                for key, value in color_state.items()
                if "batches" not in key
            }
            self.color_encoder.load_state_dict(color_state)

        self.num_encoder = get_mlp([1, 64, embed_dim])
        self.num_mean = 1826.6844940968194
        self.num_std = 2516.8905096993817
        self.pointnet = PointNet2(
            len(known_classes),
            len(known_colors),
            args,
        )
        if args.prealign_pointnet_path:
            pointnet_state = torch.load(
                args.prealign_pointnet_path,
                map_location="cpu",
            )
            pointnet_state = {
                key.split("pointnet.")[1]: value
                for key, value in pointnet_state.items()
                if "batches" not in key and "mlp_pointnet" not in key
            }
            self.pointnet.load_state_dict(pointnet_state)

        if args.pointnet_freeze:
            print("CARE: freezing PN")
            self.pointnet.requires_grad_(False)

        pointnet_dims = (
            self.pointnet.dim0,
            self.pointnet.dim1,
            self.pointnet.dim2,
        )
        self.mlp_pointnet = get_mlp(
            [pointnet_dims[args.pointnet_features], self.embed_dim]
        )
        self.mlp_merge = get_mlp(
            [len(args.use_features) * embed_dim, embed_dim]
        )

    def forward(self, objects: List[Object3d], object_points):
        if self.args.class_embed or self.args.color_embed:
            class_indices = []
            color_indices = []
            for objects_sample in objects:
                for obj in objects_sample:
                    class_indices.append(
                        self.known_classes.get(obj.label, 0)
                    )
                    color_indices.append(
                        self.known_colors.get(obj.get_color_text(), 0)
                    )

        if not self.args.class_embed:
            if "color" not in self.args.use_features:
                for pyg_batch in object_points:
                    pyg_batch.x[:] = 0.0
            object_features = [
                self.pointnet(
                    pyg_batch.to(self.get_device())
                ).features2
                for pyg_batch in object_points
            ]
            object_features = torch.cat(object_features, dim=0)
            object_features = self.mlp_pointnet(object_features)

        embeddings = []
        if "class" in self.args.use_features:
            if self.args.class_embed:
                class_embedding = self.class_embedding(
                    torch.tensor(
                        class_indices,
                        dtype=torch.long,
                        device=self.get_device(),
                    )
                )
                embeddings.append(safe_l2_normalize(class_embedding))
            else:
                embeddings.append(safe_l2_normalize(object_features))

        if "color" in self.args.use_features:
            if self.args.color_embed:
                color_embedding = self.color_embedding(
                    torch.tensor(
                        color_indices,
                        dtype=torch.long,
                        device=self.get_device(),
                    )
                )
                embeddings.append(safe_l2_normalize(color_embedding))
            else:
                colors = np.asarray(
                    [
                        obj.get_color_rgb()
                        for objects_sample in objects
                        for obj in objects_sample
                    ],
                    dtype=np.float32,
                )
                color_embedding = self.color_encoder(
                    torch.tensor(
                        colors,
                        dtype=torch.float,
                        device=self.get_device(),
                    )
                )
                color_embedding = torch.nan_to_num(
                    color_embedding,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                embeddings.append(safe_l2_normalize(color_embedding))

        if "position" in self.args.use_features:
            positions = []
            relative_distances = []
            for objects_sample in objects:
                positions.extend(
                    obj.get_center()
                    for obj in objects_sample
                )
                object_count = len(objects_sample)
                relation_matrix = np.zeros(
                    [object_count, object_count],
                    dtype=np.float32,
                )
                for row in range(object_count):
                    position = objects_sample[row].get_center()[0:2]
                    for column in range(object_count):
                        other = objects_sample[column].get_center()[0:2]
                        relation_matrix[row][column] = np.linalg.norm(
                            position - other
                        )
                relative_distances.append(
                    safe_l2_normalize(
                        torch.from_numpy(relation_matrix),
                        dim=1,
                    )
                )
            relative_position_embeddings = torch.stack(
                relative_distances,
                dim=0,
            )
            positions = np.asarray(positions, dtype=np.float32)
            position_embedding = self.pos_encoder(
                torch.tensor(
                    positions,
                    dtype=torch.float,
                    device=self.get_device(),
                )
            )
            position_embedding = torch.nan_to_num(
                position_embedding,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            embeddings.append(safe_l2_normalize(position_embedding))

        if "num" in self.args.use_features:
            num_points = np.asarray(
                [
                    len(obj.xyz)
                    for objects_sample in objects
                    for obj in objects_sample
                ],
                dtype=np.float32,
            )
            normalized_counts = (
                torch.tensor(
                    num_points,
                    dtype=torch.float,
                    device=self.get_device(),
                ).unsqueeze(-1)
                - self.num_mean
            ) / self.num_std
            normalized_counts = torch.clamp(
                normalized_counts,
                min=-10.0,
                max=10.0,
            )
            count_embedding = self.num_encoder(normalized_counts)
            count_embedding = torch.nan_to_num(
                count_embedding,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            embeddings.append(safe_l2_normalize(count_embedding))

        if len(embeddings) > 1:
            merged = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            merged = embeddings[0]
        merged = torch.nan_to_num(
            merged,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            merged,
            position_embedding,
            relative_position_embeddings,
        )

    def get_device(self):
        return next(self.class_embedding.parameters()).device

