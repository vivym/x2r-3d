from typing import Tuple

import torch


@torch.jit.script
def normalize(
    vertices: torch.Tensor, faces: torch.Tensor, mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize the vertices and faces of a mesh.

    Args:
        vertices: The vertices of shape [V, 3]
        faces: The faces of shape [F, 3]
        mode: The normalization mode. Currently only "sphere" is supported.
    """

    if mode == "sphere":
        v_max, _ = torch.max(vertices, dim=0)
        v_min, _ = torch.min(vertices, dim=0)
        v_center = (v_max + v_min) / 2.0
        vertices = vertices - v_center

        # Find the max distance to origin
        max_dist = torch.sqrt(torch.max(torch.sum(vertices ** 2, dim=-1)))
        v_scale = 1.0 / max_dist
        vertices = vertices * v_scale
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return vertices, faces
