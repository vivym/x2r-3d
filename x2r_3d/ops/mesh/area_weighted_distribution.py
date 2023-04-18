from typing import Optional

import torch
from torch.distributions.distribution import Distribution

from .per_face_normals import get_per_face_normals


@torch.jit.script
def get_areas(normals: torch.Tensor) -> torch.Tensor:
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10
    return areas


def get_area_weighted_distribution(
    vertices: torch.Tensor, faces: torch.Tensor, normals: Optional[torch.Tensor] = None
) -> Distribution:
    if normals is None:
        normals = get_per_face_normals(vertices, faces)

    areas = get_areas(normals)

    # Discrete PDF over triangles
    return torch.distributions.Categorical(areas.view(-1))
