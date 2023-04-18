from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict

import torch
import numpy as np
import tinyobjloader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

texopts = [
    "ambient_texname",
    "diffuse_texname",
    "specular_texname",
    "specular_highlight_texname",
    "bump_texname",
    "displacement_texname",
    "alpha_texname",
    "reflection_texname",
    "roughness_texname",
    "metallic_texname",
    "sheen_texname",
    "emissive_texname",
    "normal_texname"
]


@dataclass
class LoadMeshResult:
    vertices: torch.Tensor
    faces: torch.Tensor
    texture_vertices: Optional[torch.Tensor] = None
    texture_faces: Optional[torch.Tensor] = None
    materials: Optional[List[Dict[str, torch.Tensor]]] = None


def load_material(mat_path : Union[str, Path]) -> torch.Tensor:
    """Load material.
    """

    img = torch.from_numpy(np.asarray(Image.open(mat_path)))
    img = img.float() / 255.0

    return img


def load_mesh(
    mesh_path: Union[str, Path],
    load_materials: bool = False,
) -> LoadMeshResult:
    """Loads a mesh from a file.

    Args:
        mesh_path: Path to the mesh file.

    Returns:
        A tuple containing the vertices and faces of the mesh.
    """
    reader = tinyobjloader.ObjReader()
    config = tinyobjloader.ObjReaderConfig()
    config.triangulate = True   # Ensure we don't have any polygons

    ret = reader.ParseFromFile(str(mesh_path), config)

    if not ret:
        print("Failed to load : ", mesh_path)
        print("Warn:", reader.Warning())
        print("Err:", reader.Error())
        exit(-1)

    if reader.Warning():
        print("Warn:", reader.Warning())

    attrib = reader.GetAttrib()

    vertices = torch.as_tensor(attrib.vertices, dtype=torch.float32).reshape(-1, 3)

    shapes = reader.GetShapes()
    faces = []
    for shape in shapes:
        faces += [idx.vertex_index for idx in shape.mesh.indices]
    faces = torch.as_tensor(faces, dtype=torch.int64).reshape(-1, 3)

    if load_materials:
        # Load per-faced texture coordinate indices

        texf = []
        matf = []
        for shape in shapes:
            texf += [idx.texcoord_index for idx in shape.mesh.indices]
            matf += shape.mesh.material_ids
        # texf stores [tex_idx0, tex_idx1, tex_idx2, mat_idx]
        texf = torch.as_tensor(texf, dtype=torch.int64).reshape(-1, 3)
        matf = torch.as_tensor(matf, dtype=torch.int64)[:, None]
        texf = torch.cat([texf, matf], dim=-1)

        # Load texcoords
        texv = torch.as_tensor(attrib.texcoords, dtype=torch.float32).reshape(-1, 2)

        # Load texture maps
        parent_path = Path(mesh_path).parent
        materials = reader.GetMaterials()
        mats = []
        for material in materials:
            mat = {}
            diffuse = getattr(material, "diffuse")
            if diffuse:
                mat["diffuse"] = torch.as_tensor(diffuse, dtype=torch.float32)

            for texopt in texopts:
                mat_path = getattr(material, texopt)
                if mat_path:
                    img = load_material(parent_path / mat_path)
                    mat[texopt] = img
                    mat[texopt.split('_')[0]] = img

            mats.append(mat)
    else:
        texv, texf, mats = None, None, None

    return LoadMeshResult(
        vertices=vertices,
        faces=faces,
        texture_vertices=texv,
        texture_faces=texf,
        materials=mats,
    )
