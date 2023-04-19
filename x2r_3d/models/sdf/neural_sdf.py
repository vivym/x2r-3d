from dataclasses import dataclass
from typing import Optional

import torch
import lightning.pytorch as pl

from x2r_3d.models.grids import BLASGrid
from x2r_3d.models.positional_embeddings import PositionalEmbedding
from .decoder import BasicDecoder


@dataclass
class Result:
    sdf: torch.Tensor


class NeuralSDF(pl.LightningModule):
    def __init__(
        self,
        grid: Optional[BLASGrid] = None,
        position_embedding: Optional[PositionalEmbedding] = None,
    ) -> None:
        super().__init__()

        self.grid = grid
        self.position_embedding = position_embedding

        if self.grid.aggregation_type == "cat":
            feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            feature_dim = self.grid.feature_dim

        self.decoder = BasicDecoder(
            in_channels=feature_dim + self.position_embedding.out_channels,
            out_channels=1,
        )

    def forward(self, coords: torch.Tensor, lod_idx: Optional[int] = None) -> Result:
        if coords.shape[0] == 0:
            return Result(sdf=torch.zeros_like(coords[..., 0:1]))

        if lod_idx is None and hasattr(self.grid, "num_lods"):
            lod_idx = self.grid.num_lods - 1

        if coords.dim() == 2:
            coords = coords[:, None]

        feats = self.grid.interpolate(coords, lod_idx)

        pos_embeds = self.position_embedding(coords)
        if pos_embeds is not None:
            feats = torch.cat([pos_embeds, feats], dim=-1)

        sdf, _ = self.decoder(feats)

        if coords.dim() == 2:
            sdf = sdf.squeeze(1)

        return Result(sdf=sdf)

    def training_step(self, batch, _) -> torch.Tensor:
        ...

    def validation_step(self, batch, _) -> None:
        ...
