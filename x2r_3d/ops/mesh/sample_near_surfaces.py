from typing import Optional

import torch
from torch.distributions.distribution import Distribution

from .sample_surfaces import sample_surfaces


def sample_near_surfaces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    variance: float = 0.01,
    distrib: Optional[Distribution] = None,
) -> torch.Tensor:
    samples, _ = sample_surfaces(vertices, faces, num_samples, distrib)

    # Randomly perturb the samples
    perturbations = torch.randn_like(samples) * variance
    samples += perturbations

    return samples
