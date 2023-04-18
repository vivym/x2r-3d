from pathlib import Path
from typing import Union, Optional

import torch
import kaolin.ops.spc as spc_ops

from x2r_3d.ops.mesh import load_mesh, normalize
from x2r_3d.ops.spc import octree_to_spc, mesh_to_octree
from .base_as import BaseAS, ASQueryResults


class OctreeAS(BaseAS):
    def __init__(self, octree: torch.ByteTensor) -> None:
        super().__init__()

        self.octree = octree
        self.points, self.pyramid, self.prefix = octree_to_spc(octree)
        self.max_level = self.pyramid.shape[-1] - 2
        self.extra_fields = {}

        print("octree", octree.shape)
        print("points", self.points.shape)
        print("pyramid", self.pyramid.shape)
        print("prefix", self.prefix.shape)

    @classmethod
    def from_mesh(
        cls,
        mesh_path: Union[str, Path],
        max_level: int,
        use_texture: bool = False,
        num_samples: int = 100_000_000,
    ) -> "OctreeAS":
        mesh = load_mesh(mesh_path, load_materials=use_texture)

        vertices, faces = normalize(mesh.vertices.cuda(), mesh.faces.cuda(), "sphere")

        # Note: This function is not deterministic since it relies on sampling.
        # Eventually this will be replaced by 3D rasterization.
        octree = mesh_to_octree(vertices, faces, max_level, num_samples)
        accel_struct = OctreeAS(octree)
        accel_struct.extra_fields["vertices"] = vertices
        accel_struct.extra_fields["faces"] = faces

        if use_texture:
            accel_struct.extra_fields["texture_vertices"] = mesh.texture_vertices
            accel_struct.extra_fields["texture_faces"] = mesh.texture_faces
            accel_struct.extra_fields["materials"] = mesh.materials

        return accel_struct

    def query(
        self, coords: torch.Tensor, level: Optional[int] = None, with_parents: bool = False
    ) -> ASQueryResults:
        if level is None:
            level = self.max_level

        pidx = spc_ops.unbatched_query(self.octree, self.prefix, coords, level, with_parents)
        return ASQueryResults(pidx)
