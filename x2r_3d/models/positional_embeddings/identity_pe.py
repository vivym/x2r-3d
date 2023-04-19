import torch

from .positional_embedding import PositionalEmbedding


class IdentityPositionalEmbedding(PositionalEmbedding):
    @property
    def out_channels(self) -> int:
        return self.in_channels
