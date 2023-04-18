from typing import Tuple

import torch
import kaolin.ops.spc as spc_ops


def make_trilinear_spc(
    points: torch.ShortTensor, pyramid: torch.LongTensor
) -> Tuple[torch.ShortTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Builds a trilinear spc from a regular spc.

    Args:
        points (torch.ShortTensor): The point_hierarchy.
        pyramid (torch.LongTensor): The pyramid tensor.

    Returns:
        (torch.ShortTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor)
        - The dual point_hierarchy.
        - The dual pyramid.
        - The trinkets.
        - The parent pointers.
    """
    points_dual, pyramid_dual = spc_ops.unbatched_make_dual(points, pyramid)
    trinkets, parents = spc_ops.unbatched_make_trinkets(points, pyramid, points_dual, pyramid_dual)
    return points_dual, pyramid_dual, trinkets, parents
