import torch
import kaolin.ops.spc as spc_ops

from x2r_3d.ops.mesh import sample_on_surface


def octree_to_spc(octree: torch.ByteTensor) -> torch.Tensor:
    lengths = torch.as_tensor([len(octree)], dtype=torch.int32)
    _, pyramid, prefix = spc_ops.scan_octrees(octree, lengths)
    points = spc_ops.generate_points(octree, pyramid, prefix)
    return points, pyramid[0], prefix


@torch.jit.script
def quantize_points(x: torch.Tensor, level: int) -> torch.Tensor:
    r"""Quantize :math:`[-1, 1]` float coordinates in to
    :math:`[0, (2^{level})-1]` integer coords.

    If a point is out of the range :math:`[-1, 1]` it will be clipped to it.

    Args:
        x (torch.Tensor): Floating point coordinates,
                          must be of last dimension 3.
        level (int): Level of the grid

    Returns
        (torch.ShortTensor): Quantized 3D points, of same shape than x.
    """
    res = 2 ** level
    qpts = torch.floor(torch.clamp(res * (x + 1.0) / 2.0, 0, res - 1.)).short()
    return qpts


@torch.jit.script
def _augment_and_quantize_samples(sample_points: torch.Tensor, max_level: int):
    noises = (torch.rand_like(sample_points) * 2.0 - 1.0) * (1.0 / (2 ** (max_level + 1)))
    sample_points = torch.cat([sample_points, sample_points + noises], dim=0)
    sample_points = quantize_points(sample_points, max_level)
    return sample_points


def mesh_to_octree(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_level: int,
    num_samples: int = 100_000_000,
) -> torch.ByteTensor:
    sample_points = sample_on_surface(vertices, faces, num_samples)[0]
    # Augment samples... may be a hack that isn't actually needed
    sample_points = _augment_and_quantize_samples(sample_points, max_level)
    return spc_ops.unbatched_points_to_octree(sample_points, max_level)
