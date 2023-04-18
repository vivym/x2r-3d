from typing import Optional

import torch
from torch.distributions.distribution import Distribution

from .area_weighted_distribution import get_area_weighted_distribution
from .per_face_normals import get_per_face_normals


def random_faces(
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

    indices = distrib.sample([num_samples])
    assert indices.device == faces.device

    return faces[indices], normals[indices]
