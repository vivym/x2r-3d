import tempfile
from typing import Any, Dict, List
from pathlib import Path

import ray
import ray.data
import torch
import zstandard as zstd
from tqdm import tqdm

from x2r_3d.accelstructs.octree_as import OctreeAS

synset_id_to_category = {
    "02691156": "airplane", "02773838": "bag", "02801938": "basket",
    "02808440": "bathtub", "02818832": "bed", "02828884": "bench",
    "02876657": "bottle", "02880940": "bowl", "02924116": "bus",
    "02933112": "cabinet", "02747177": "can", "02942699": "camera",
    "02954340": "cap", "02958343": "car", "03001627": "chair",
    "03046257": "clock", "03207941": "dishwasher", "03211117": "monitor",
    "04379243": "table", "04401088": "telephone", "02946921": "tin_can",
    "04460130": "tower", "04468005": "train", "03085013": "keyboard",
    "03261776": "earphone", "03325088": "faucet", "03337140": "file",
    "03467517": "guitar", "03513137": "helmet", "03593526": "jar",
    "03624134": "knife", "03636649": "lamp", "03642806": "laptop",
    "03691459": "speaker", "03710193": "mailbox", "03759954": "microphone",
    "03761084": "microwave", "03790512": "motorcycle", "03797390": "mug",
    "03928116": "piano", "03938244": "pillow", "03948459": "pistol",
    "03991062": "pot", "04004475": "printer", "04074963": "remote_control",
    "04090263": "rifle", "04099429": "rocket", "04225987": "skateboard",
    "04256520": "sofa", "04330267": "stove", "04530566": "vessel",
    "04554684": "washer", "02992529": "cellphone",
    "02843684": "birdhouse", "02871439": "bookshelf",
    # "02858304": "boat", no boat in our dataset, merged into vessels
    # "02834778": "bicycle", not in our taxonomy
}
synset_ids = sorted(synset_id_to_category.keys())
category_to_category_id = {synset_id_to_category[synset_id]: i for i, synset_id in enumerate(synset_ids)}
category_to_synset_id = {v: k for k, v in synset_id_to_category.items()}


def convert_to_octrees(root_path: Path, categories: List[str]):
    items = []

    for category in categories:
        synset_id = category_to_synset_id[category]
        category_root_path = root_path / synset_id
        for obj_path in tqdm(sorted(category_root_path.glob("*"))):
            octree_path = Path(f"data/ShapeNetCoreV2-octree/{obj_path.name}.pth")
            if octree_path.exists():
                continue
            mesh_path = obj_path / "models" / "model_normalized_watertight.obj.zst"
            if not mesh_path.exists():
                continue
            # uncompress to temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                decompressed_mesh_path = Path(tmpdir) / "mesh.obj"
                with open(mesh_path, "rb") as in_f, open(decompressed_mesh_path, "wb") as out_f:
                    cctx = zstd.ZstdDecompressor()
                    cctx.copy_stream(in_f, out_f)

                # convert to octree
                blas = OctreeAS.from_mesh(decompressed_mesh_path, max_level=7)
                octree = blas.octree.cpu().numpy()
                vertices = blas.extra_fields["vertices"].cpu().numpy()
                faces = blas.extra_fields["faces"].cpu().numpy()

                torch.save(
                    dict(
                        category=category,
                        category_id=category_to_category_id[category],
                        synset_id=synset_id,
                        object_id=obj_path.name,
                        mesh_path=str(mesh_path.relative_to(root_path)),
                        octree=octree,
                        vertices=vertices,
                        faces=faces,
                    ),
                    octree_path,
                )

    #     items += [
    #         {
    #             "category": category,
    #             "category_id": category_to_category_id[category],
    #             "synset_id": synset_id,
    #             "object_id": obj_path.name,
    #             "mesh_path": str((obj_path / "models" / "model_normalized.obj").relative_to(root_path)),
    #         }
    #         for obj_path in category_root_path.glob("*")
    #     ]

    # def to_octree(data: Dict[str, Any]):
    #     mesh_path = root_path / data["mesh_path"]
    #     try:
    #         blas = OctreeAS.from_mesh(mesh_path, max_level=7)
    #     except Exception as e:
    #         return None

    #     octree = blas.octree.cpu().numpy()
    #     vertices = blas.extra_fields["vertices"].cpu().numpy()
    #     faces = blas.extra_fields["faces"].cpu().numpy()

    #     return dict(**data, octree=octree, vertices=vertices, faces=faces)

    # for item in tqdm(items):
    #     item = to_octree(item)
    #     if item:
    #         torch.save(item, f"data/shapenetcore_v2_octrees/{item['object_id']}.pth")

    # ds = ray.data.from_items(items)
    # ds = ds.repartition(64, shuffle=True)
    # ds = ds.map(to_octree, num_gpus=1)
    # ds.write_parquet(
    #     "data/shapenetcore_v2_octrees.parquet",
    #     compression="zstd",
    # )


if __name__ == "__main__":
    convert_to_octrees(
        Path("data/ShapeNetCore.v2"),
        ["car", "chair", "airplane"],
    )
