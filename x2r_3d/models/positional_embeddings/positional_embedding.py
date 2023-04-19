import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels

    @property
    def out_channels(self) -> int:
        raise NotImplementedError("out_channels not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
