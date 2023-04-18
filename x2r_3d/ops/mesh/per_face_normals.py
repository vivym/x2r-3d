import torch


@torch.jit.script
def get_per_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    mesh = vertices[faces]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    return torch.cross(vec_a, vec_b)
