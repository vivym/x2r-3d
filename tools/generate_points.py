import time
import pickle
from pathlib import Path

import ray
import torch
import numpy as np
import kaolin.ops.spc as spc_ops
import zstandard as zstd
from ray.util import ActorPool
from tqdm import tqdm


@ray.remote(num_cpus=2, num_gpus=0.5)
class PointSampler:
    def sample(self, octree_path: Path) -> int:
        from x2r_3d.ops.mesh import (
            sample_on_surface,
            sample_near_surface,
            get_per_face_normals,
            get_area_weighted_distribution,
        )
        from x2r_3d.ops.spc import sample_uniform_spc

        sample_modes = ["rand", "near", "near", "trace", "trace"]
        samples_per_voxel = 32

        # start_time = time.perf_counter()

        with open(octree_path, "rb") as f:
            data = pickle.load(f)

        vertices = torch.from_numpy(data["mesh_vertices"]).cuda(non_blocking=True)
        faces = torch.from_numpy(data["mesh_faces"]).cuda(non_blocking=True)
        level = data["octree_max_level"]
        octree = torch.from_numpy(data["octree"]).cuda(non_blocking=True)
        octree_prefix = torch.from_numpy(data["octree_prefix"]).cuda(non_blocking=True)
        octree_points = torch.from_numpy(data["octree_points"]).cuda(non_blocking=True)
        octree_pyramid = torch.from_numpy(data["octree_pyramid"]).cuda(non_blocking=True)

        # torch.cuda.synchronize()

        # print(f"Loading octree took {time.perf_counter() - start_time:.2f} seconds")

        # start_time = time.perf_counter()

        # Here, corners mean "the bottom left corner of the voxel to sample from"
        corners = spc_ops.unbatched_get_level_points(octree_points, octree_pyramid, level)

        # Two pass sampling to figure out sample size
        pts = []
        for mode in sample_modes:
            if mode == "rand":
                pts.append(sample_uniform_spc(corners, level, samples_per_voxel))

        normals = get_per_face_normals(vertices, faces)
        distrib = get_area_weighted_distribution(vertices, faces, normals)

        num_sample_points = pts[0].shape[0]
        for mode in sample_modes:
            if mode == "rand":
                ...
            elif mode == "near":
                pts.append(
                    sample_near_surface(
                        vertices,
                        faces,
                        num_sample_points,
                        variance=1.0 / (2 ** level),
                        normals=normals,
                        distrib=distrib,
                    )
                )
            elif mode == "trace":
                pts.append(
                    sample_on_surface(
                        vertices, faces, num_sample_points, normals=normals, distrib=distrib
                    )[0]
                )
            else:
                raise Exception(f"Sampling mode {mode} not implemented")

        # Filter out points which do not belong to the narrowband
        pts = torch.cat(pts, dim=0)
        pidx = spc_ops.unbatched_query(octree, octree_prefix, pts, level=0)
        pts = pts[pidx > -1]
        assert pts.is_cuda
        pts = pts.cpu().numpy()

        # print(f"Sampled {pts.shape[0]} points in {time.perf_counter() - start_time:.2f} seconds.")

        with open(octree_path.parent.parent / "pts" / (octree_path.stem + ".pkl"), "wb") as f:
            data = pickle.dumps(pts)
            f.write(data)

        return pts.shape[0]


def main():
    root_path = Path("data/ShapeNetCoreV2/octrees.cache")
    octree_paths = sorted(root_path.glob("*"))

    pool = ActorPool([PointSampler.remote() for _ in range(6)])

    num_points_list = pool.map(lambda sampler, v: sampler.sample.remote(v), octree_paths)
    num_points_list = list(tqdm(num_points_list, total=len(octree_paths)))
    with open(root_path.parent / "num_points.pkl", "wb") as f:
        pickle.dump(num_points_list, f)

    print("num_points_list", np.min(num_points_list), np.max(num_points_list), np.mean(num_points_list), np.std(num_points_list))


if __name__ == "__main__":
    main()
