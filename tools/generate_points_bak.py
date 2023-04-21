import ray.data
import torch
import kaolin.ops.spc as spc_ops
from ray.data import ActorPoolStrategy
from ray.data.extensions import ArrowVariableShapedTensorType

from x2r_3d.ops.mesh import (
    sample_on_surface,
    sample_near_surface,
    get_per_face_normals,
    get_area_weighted_distribution,
)
from x2r_3d.ops.spc import sample_uniform_spc


def sample_points(data):
    sample_modes = ["rand", "near", "near", "trace", "trace"]
    samples_per_voxel = 32

    vertices = torch.from_numpy(data["mesh_vertices"]).cuda(non_blocking=True)
    faces = torch.from_numpy(data["mesh_faces"]).cuda(non_blocking=True)
    level = torch.from_numpy(data["octree_max_level"]).cuda(non_blocking=True)
    octree = torch.from_numpy(data["octree"]).cuda(non_blocking=True)
    octree_prefix = torch.from_numpy(data["octree_prefix"]).cuda(non_blocking=True)
    octree_points = torch.from_numpy(data["octree_points"]).cuda(non_blocking=True)
    octree_pyramid = torch.from_numpy(data["octree_pyramid"]).cuda(non_blocking=True)

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

    return data
    # return dict(
    #     **data,
    #     pts=pts.cpu().numpy(),
    # )


def main():
    ds = ray.data.read_parquet("data/ShapeNetCoreV2/octrees.parquet")
    ds.schema()
    print(ds)
    # print(ds.schema())
    # ds = ds.map(sample_points, compute=ActorPoolStrategy(4, 6), num_gpus=0.5, num_cpus=6)
    # ds.write_parquet("data/ShapeNetCoreV2/pts.parquet")


if __name__ == "__main__":
    main()
