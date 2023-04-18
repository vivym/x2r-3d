import torch
import kaolin.ops.spc as spc_ops

from x2r_3d.ops.mesh import sample_surfaces


def octree_to_spc(octree: torch.ByteTensor) -> torch.Tensor:
    lengths = torch.as_tensor([len(octree)], dtype=torch.int32)
    _, pyramid, prefix = spc_ops.scan_octrees(octree, lengths)
    points = spc_ops.generate_points(octree, pyramid, prefix)
    return points, pyramid[0], prefix


def mesh_to_octree(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_level: int,
    num_samples: int = 100_000_000,
) -> torch.ByteTensor:
    sample_points = sample_surfaces(vertices, faces, num_samples)[0]
    # TODO: torch.jit.script here
    # Augment samples... may be a hack that isn't actually needed
    noises = (torch.rand_like(sample_points) * 2.0 - 1.0) * (1.0 / (2 ** (max_level + 1)))
    sample_points = torch.cat([sample_points, sample_points + noises], dim=0)
    sample_points = spc_ops.quantize_points(sample_points, max_level)
    return spc_ops.unbatched_points_to_octree(sample_points, max_level)
