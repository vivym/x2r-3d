from typing import Optional, List, Callable

from torch.utils.data import Dataset

from x2r_3d.accelstructs.octree_as import OctreeAS


class OctreeSDFDataset(Dataset):
    def __init__(
        self,
        blas: OctreeAS,
        transform: Optional[Callable] = None,
        sample_mode: Optional[List[str]] = None,
        num_samples: int = 100_000,
        use_texture: bool = False,
        num_samples_per_voxel: int = 32,
    ) -> None:
        super().__init__()

        self.blas = blas
        self.transform = transform

        if sample_mode is None:
            sample_mode = ["rand", "rand", "near", "near", "trace"]
        self.sample_mode = sample_mode
        self.num_samples = num_samples
        self.use_texture = use_texture
        self.num_samples_per_voxel = num_samples_per_voxel

        assert "vertices" in blas.extra_fields and "faces" in blas.extra_fields

        self._load()

    def _load(self):
        ...
