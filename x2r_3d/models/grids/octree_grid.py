import torch
import torch.nn as nn
import kaolin.ops.spc as spc_ops

from x2r_3d.accelstructs.octree_as import OctreeAS
from x2r_3d.ops.spc import make_trilinear_spc
from .blas_grid import BLASGrid


class OctreeGrid(BLASGrid):
    def __init__(
        self,
        accelstruct: OctreeAS,
        feature_dim: int,
        base_lod: int = 2,
        num_lods: int = 6,
        interpolation_type: str = "linear",
        aggregation_type: str = "cat",
        feature_std: float = 0.01,
        feature_bias: float = 0.0,
    ) -> None:
        super().__init__(accelstruct)

        self.feature_dim = feature_dim
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.aggregation_type = aggregation_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias

        # List of octree levels which are optimized.
        self.active_lods = [self.base_lod + x for x in range(self.num_lods)]
        self.max_lod = base_lod + num_lods - 1

        if self.num_lods > 0:
            self._init_features()

    def _init_features(self):
        # Build the pyramid of features
        fpyramid = []
        for al in self.active_lods:
            if self.interpolation_type == "linear":
                fpyramid.append(self.blas.pyramid_dual[0, al].item() + 1)
            elif self.interpolation_type == "closest":
                fpyramid.append(self.blas.pyramid[0, al].item() + 1)
            else:
                raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")

        self.num_feats = sum(fpyramid)

        self.features = nn.ParameterList([])
        for i in range(len(self.active_lods)):
            fts = torch.zeros(fpyramid[i], self.feature_dim) + self.feature_bias
            fts += torch.randn_like(fts) * self.feature_std
            self.features.append(nn.Parameter(fts))

    def freeze(self):
        """Freezes the feature grid.
        """
        for lod_idx in range(self.num_lods):
            self.features[lod_idx].requires_grad_(False)

    def interpolate(self, coords: torch.Tensor, lod_idx: int) -> torch.Tensor:
        output_shape = coords.shape[:-1]
        if coords.dim() == 2:
            # b, 3 -> b, n, 3
            coords = coords[:, None]

        if lod_idx == 0:
            query_results = self.blas.query(coords[:,0], self.active_lods[lod_idx], with_parents=False)
            pidx = query_results.pidx
            feat = self._interpolate(coords, self.features[0], pidx, 0)
            # return feat.reshape(*output_shape, feat.shape[-1])

            query_results = self.blas.query(
                coords[:, 0], self.active_lods[lod_idx], with_parents=False
            )
            pidx = query_results.pidx
            feats = self._interpolate(coords, self.features[0], pidx, 0)
            return feats.reshape(*output_shape, -1)
        else:
            feats = []

            # In the multiscale case, the raytrace _currently_  does not return multiscale indices.
            # As such, no matter what the pidx will be recalculated to get the multiscale indices.
            num_feats = lod_idx + 1

            # This might look unoptimal since it assumes that samples are _not_ in the same voxel.
            # This is the correct assumption here, because the point samples are from the base_lod,
            # not the highest LOD.
            query_results = self.blas.query(
                coords.reshape(-1, 3), self.active_lods[lod_idx], with_parents=True
            )
            pidx = query_results.pidx[..., self.base_lod:]
            pidx = pidx.reshape(-1, coords.shape[1], num_feats)
            pidx = torch.split(pidx, 1, dim=-1)

            for i in range(num_feats):
                feat = self._interpolate(
                    coords.reshape(-1, 1, 3), self.features[i], pidx[i].reshape(-1), i)[:, 0]
                feats.append(feat)

            feats = torch.cat(feats, dim=-1)

            if self.aggregation_type == "sum":
                feats = feats.reshape(*feats.shape[:-1], num_feats, self.feature_dim)
                feats = feats.sum(-2)

            return feats.reshape(*output_shape, -1)

    def _interpolate(
        self, coords: torch.Tensor, feats: torch.Tensor, pidx: torch.LongTensor, lod_idx: int
    ) -> torch.Tensor:
        lod = self.active_lods[lod_idx]

        if self.interpolation_type == "linear":
            feats = spc_ops.unbatched_interpolate_trilinear(
                coords, pidx.int(), self.blas.points, self.blas.trinkets.int(), feats.half(), lod
            ).to(dtype=feats.dtype)
        elif self.interpolation_type == "closest":
            assert False
        else:
            raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")

        return feats
