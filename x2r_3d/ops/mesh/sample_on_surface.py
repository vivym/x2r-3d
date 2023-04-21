from typing import Optional

import torch
from torch.distributions.distribution import Distribution

from .area_weighted_distribution import get_area_weighted_distribution
from .per_face_normals import get_per_face_normals
from .random_faces import random_faces


@torch.jit.script
def get_samples(f: torch.Tensor, num_samples: int) -> torch.Tensor:
    u = torch.sqrt(torch.rand(num_samples, device=f.device))[..., None]
    v = torch.rand(num_samples, device=f.device)[..., None]

    return (1 - u) * f[:, 0, :] + (u * (1 - v)) * f[:, 1, :] + u * v * f[:, 2, :]


def sample_on_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    normals: Optional[torch.Tensor] = None,
    distrib: Optional[Distribution] = None,
):
    if normals is None:
        normals = get_per_face_normals(vertices, faces)

    if distrib is None:
        distrib = get_area_weighted_distribution(vertices, faces, normals)

    faces, normals = random_faces(vertices, faces, num_samples, normals, distrib)
    f = vertices[faces]

    samples = get_samples(f, num_samples)

    return samples, normals
