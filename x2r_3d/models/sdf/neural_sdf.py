from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from x2r_3d.models.grids import BLASGrid
from x2r_3d.models.positional_embeddings import PositionalEmbedding, IdentityPositionalEmbedding
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

        if position_embedding is None:
            # TODO: support none positional embedding
            position_embedding = IdentityPositionalEmbedding(3)

        self.position_embedding = position_embedding

        if self.grid.aggregation_type == "cat":
            feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            feature_dim = self.grid.feature_dim

        self.decoder = BasicDecoder(
            in_channels=feature_dim + self.position_embedding.out_channels,
            out_channels=1,
        )

        self.loss_lods = list(range(self.grid.num_lods))

    def forward(self, coords: torch.Tensor, lod_idx: Optional[int] = None) -> Result:
        if coords.shape[0] == 0:
            return Result(sdf=torch.zeros_like(coords[..., 0:1]))

        output_shape = coords.shape[:-1]

        if lod_idx is None and hasattr(self.grid, "num_lods"):
            lod_idx = self.grid.num_lods - 1

        if coords.dim() == 2:
            coords = coords[:, None]

        feats = self.grid.interpolate(coords, lod_idx)

        pos_embeds = self.position_embedding(coords)
        if pos_embeds is not None:
            feats = torch.cat([pos_embeds, feats], dim=-1)

        sdf, _ = self.decoder(feats)

        return Result(sdf=sdf.reshape(*output_shape))

    def training_step(self, batch, _) -> torch.Tensor:
        coords, sdf = batch["coords"], batch["sdf"]

        loss = 0.
        for lod_idx in self.loss_lods:
            pred_sdf = self(coords, lod_idx=lod_idx).sdf
            loss += F.mse_loss(pred_sdf, sdf)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, _) -> None:
        ...
