import time
import pickle
from pathlib import Path

import ray
import torch
import numpy as np
from ray.util import ActorPool
from tqdm import tqdm
from safetensors.numpy import save_file


@ray.remote(num_cpus=2, num_gpus=0.5)
class PointSampler:
    def sample(self, octree_path: Path) -> int:
        from x2r_3d.ops.spc import make_trilinear_spc

        with open(octree_path, "rb") as f:
            data = pickle.load(f)

        octree_points = torch.from_numpy(data["octree_points"]).cuda(non_blocking=True)
        octree_pyramid = torch.from_numpy(data["octree_pyramid"]).cuda(non_blocking=True)

        points_dual, pyramid_dual, trinkets, parents = \
            make_trilinear_spc(octree_points, octree_pyramid)

        data["octree_points_dual"] = points_dual.cpu().numpy()
        data["octree_pyramid_dual"] = pyramid_dual.cpu().numpy()
        data["octree_trinkets"] = trinkets.cpu().numpy()
        data["octree_parents"] = parents.cpu().numpy()

        output_path = octree_path.parent.parent / "octrees" / (octree_path.name + ".pkl")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)


def main():
    root_path = Path("data/ShapeNetCoreV2/octrees.cache")
    octree_paths = sorted(root_path.glob("*"))

    pool = ActorPool([PointSampler.remote() for _ in range(6)])

    res = pool.map(lambda sampler, v: sampler.sample.remote(v), octree_paths)
    list(tqdm(res, total=len(octree_paths)))


if __name__ == "__main__":
    main()
