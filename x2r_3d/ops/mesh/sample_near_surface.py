from typing import Optional

import torch
from torch.distributions.distribution import Distribution

from .sample_on_surface import sample_on_surface


def sample_near_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    variance: float = 0.01,
    normals: Optional[torch.Tensor] = None,
    distrib: Optional[Distribution] = None,
) -> torch.Tensor:
    samples, _ = sample_on_surface(
        vertices, faces, num_samples, normals=normals, distrib=distrib
    )

    # Randomly perturb the samples
    perturbations = torch.randn_like(samples) * variance
    samples += perturbations

    return samples
