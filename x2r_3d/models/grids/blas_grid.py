from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from x2r_3d.accelstructs import BaseAS


class BLASGrid(nn.Module, metaclass=ABCMeta):
    """
    BLAS: "Bottom Level Acceleration Structure"
    """

    def __init__(self, blas: BaseAS) -> None:
        super().__init__()

        self.blas = blas

    @abstractmethod
    def interpolate(self, coords: torch.Tensor, lod_idx: Optional[int] = None) -> torch.Tensor:
        ...
