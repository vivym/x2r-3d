import torch


@torch.jit.script
def sample_uniform_spc(corners: torch.Tensor, level: int, num_samples_per_voxel: int) -> torch.Tensor:
    res = 2.0 ** level

    samples = torch.rand(corners.shape[0], num_samples_per_voxel, 3, device=corners.device)
    samples = corners[..., None, :3] + samples
    samples = samples.reshape(-1, 3)
    samples /= res

    return samples * 2.0 - 1.0
