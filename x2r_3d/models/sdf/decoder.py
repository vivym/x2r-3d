from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int = 128,
        num_layers: int = 1,
        bias: bool = True,
        skip: Optional[List[int]] = None
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.bias = bias

        if skip is None:
            skip = []
        self.skip = skip

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_channels, num_channels, bias=bias))
            elif i in self.skip:
                layers.append(nn.Linear(in_channels + num_channels, num_channels, bias=bias))
            else:
                layers.append(nn.Linear(num_channels, num_channels, bias=bias))

        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(num_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        for i, layer in enumerate(self.layers):
            x = F.relu_(layer(x))
            if i in self.skip:
                x = torch.cat([shortcut, x], dim=-1)

        return self.out_proj(x), x
