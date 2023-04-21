import pickle
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import kaolin.ops.spc as spc_ops

from x2r_3d.ops.mesh import load_mesh, normalize
from x2r_3d.ops.spc import octree_to_spc, mesh_to_octree
from .base_as import BaseAS, ASQueryResults


class OctreeAS(BaseAS):
    def __init__(self, octree_path: Path) -> None:
        super().__init__()

        if octree_path.suffix == ".pkl":
            with open(octree_path, "rb") as f:
                octree_dict = pickle.load(f)
        else:
            raise ValueError(f"Unknown file type {octree_path.suffix}")

        self.register_buffer("octree", torch.from_numpy(octree_dict["octree"]))
        self.register_buffer("points", torch.from_numpy(octree_dict["octree_points"]))
        self.register_buffer("pyramid", torch.from_numpy(octree_dict["octree_pyramid"]))
        self.register_buffer("prefix", torch.from_numpy(octree_dict["octree_prefix"]))
        self.register_buffer("points_dual", torch.from_numpy(octree_dict["octree_points_dual"]))
        self.register_buffer("pyramid_dual", torch.from_numpy(octree_dict["octree_pyramid_dual"]))
        self.register_buffer("trinkets", torch.from_numpy(octree_dict["octree_trinkets"]))
        self.register_buffer("parents", torch.from_numpy(octree_dict["octree_parents"]))
        self.register_buffer("vertices", torch.from_numpy(octree_dict["mesh_vertices"]))
        self.register_buffer("faces", torch.from_numpy(octree_dict["mesh_faces"]))

    def query(
        self, coords: torch.Tensor, level: Optional[int] = None, with_parents: bool = False
    ) -> ASQueryResults:
        if level is None:
            level = self.max_level

        pidx = spc_ops.unbatched_query(self.octree, self.prefix, coords, level, with_parents)
        return ASQueryResults(pidx)
