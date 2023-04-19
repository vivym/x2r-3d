from typing import Optional, List, Callable

import torch
import kaolin.ops.spc as spc_ops
from torch.utils.data import Dataset
from wisp.ops.mesh import compute_sdf

from x2r_3d.accelstructs.octree_as import OctreeAS
from x2r_3d.ops.spc import sample_spc
from x2r_3d.ops.mesh import (
    sample_near_surfaces, sample_surfaces, get_area_weighted_distribution
)


class OctreeSDFDataset(Dataset):
    def __init__(
        self,
        octree_path: str,
        transform: Optional[Callable] = None,
        sample_mode: Optional[List[str]] = None,
        num_samples: int = 100_000,
        use_texture: bool = False,
        num_samples_per_voxel: int = 32,
    ) -> None:
        super().__init__()

        self.octree_path = octree_path

        self.blas = OctreeAS.from_dict(octree_dict=torch.load(octree_path))
        self.transform = transform

        if sample_mode is None:
            sample_mode = ["rand", "rand", "near", "near", "trace"]
        self.sample_mode = sample_mode
        self.num_samples = num_samples
        self.use_texture = use_texture
        self.num_samples_per_voxel = num_samples_per_voxel

        assert "vertices" in self.blas.extra_fields and "faces" in self.blas.extra_fields

        self.data_pool = self._sample_from_grid()

    def _sample_from_grid(self):
        # TODO (operel): TBD when kaolin adds a mesh class:
        #  grid is only really needed for filtering out points and more efficient 'rand',
        #  better give the mesh as another input and not store the mesh contents in the extent field
        vertices = self.blas.extent["vertices"].cuda()
        faces = self.blas.extent["faces"].cuda()
        level = self.blas.max_level

        # Here, corners mean "the bottom left corner of the voxel to sample from"
        corners = spc_ops.unbatched_get_level_points(
            self.blas.points.cuda(), self.blas.pyramid.cuda(), level
        )

        # Two pass sampling to figure out sample size
        pts = []
        for mode in self.sample_mode:
            if mode == "rand":
                # Sample the points.
                pts.append(sample_spc(corners, level, self.num_samples_per_voxel))

        distrib = get_area_weighted_distribution(vertices, faces)
        for mode in self.sample_mode:
            if mode == "rand":
                pass
            elif mode == "near":
                pts.append(
                    sample_near_surfaces(
                        vertices,
                        faces,
                        pts[0].shape[0],
                        variance=1.0 / (2 ** level),
                        distrib=distrib,
                    )
                )
            elif mode == "trace":
                pts.append(sample_surfaces(vertices, faces, pts[0].shape[0], distrib)[0])
            else:
                raise Exception(f"Sampling mode {mode} not implemented")

        # Filter out points which do not belong to the narrowband
        pts = torch.cat(pts, dim=0)
        query_results = self.blas.query(pts, 0)
        pts = pts[query_results.pidx > -1]

        d = compute_sdf(vertices, faces, pts)

        return dict(
            coords=pts.cpu(),
            sdf=d.cpu(),
        )
