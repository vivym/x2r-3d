from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ASQueryResults:
    pidx: torch.LongTensor
    """ Holds the query results.
    - If the query is invoked with `with_parents=False`, this field is a tensor of shape [num_coords],
      containing indices of cells of the acceleration structure, where the query coordinates match.
    - If the query is invoked with `with_parents=True`, this field is a tensor of shape [num_coords, level+1],
      containing indices of the cells of the acceleration structure + the full parent hierarchy of each
      cell query result.
    """


class BaseAS(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def query(
        self, coords: torch.Tensor, level: Optional[int] = None, with_parents: bool = False
    ) -> ASQueryResults:
        ...
