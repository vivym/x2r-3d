from dataclasses import dataclass
from typing import Optional

import torch
import lightning.pytorch as pl

from x2r_3d.models.grids import BLASGrid
from .decoder import BasicDecoder


@dataclass
class Result:
    sdf: torch.Tensor


class NeuralSDF(pl.LightningModule):
    def __init__(
        self,
        grid: Optional[BLASGrid] = None,
    ) -> None:
        super().__init__()

        self.grid = grid

        self.decoder = BasicDecoder()

    def forward(self, coords: torch.Tensor, lod_idx: Optional[int] = None) -> Result:
        if coords.shape[0] == 0:
            return Result(sdf=torch.zeros_like(coords[..., 0:1]))

        if lod_idx is None and hasattr(self.grid, "num_lods"):
            lod_idx = self.grid.num_lods - 1

        if coords.dim() == 2:
            coords = coords[:, None]

        feats = self.grid.interpolate(coords, lod_idx)

        # TODO: Optionally concat the positions to the embedding

        sdf, _ = self.decoder(feats)

        if coords.dim() == 2:
            sdf = sdf.squeeze(1)

        return Result(sdf=sdf)
