import pickle
from typing import Any, Dict, List
from pathlib import Path

import ray
import ray.data
import numpy as np
from ray.data import ActorPoolStrategy
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
    items: List[Dict[str, Any]] = []

    for category in categories:
        synset_id = category_to_synset_id[category]
        category_root_path = root_path / synset_id
        for obj_path in tqdm(sorted(category_root_path.glob("*"))):
            mesh_path = obj_path / "models" / "model_normalized_watertight.obj"
            if not mesh_path.exists():
                continue

            items.append({
                "category": category,
                "category_id": category_to_category_id[category],
                "synset_id": synset_id,
                "object_id": obj_path.name,
                "mesh_path": str(mesh_path.relative_to(root_path)),
            })

    def to_octree(item: Dict[str, Any]) -> Dict[str, Any]:
        object_id = item["object_id"]
        cache_path = Path("data/ShapeNetCoreV2/octrees.cache") / object_id

        # if cache_path.exists():
        #     load_from_cache = True
        #     with open(cache_path, "rb") as f:
        #         return pickle.load(f)
        # else:
        #     load_from_cache = False

        try:
            blas = OctreeAS.from_mesh(root_path / item["mesh_path"], max_level=7)
            octree = blas.octree.cpu().numpy()
            points = blas.points.cpu().numpy()
            pyramid = blas.pyramid.cpu().numpy()
            prefix = blas.prefix.cpu().numpy()
            max_level = blas.max_level
            vertices = blas.extra_fields["vertices"].cpu().numpy()
            faces = blas.extra_fields["faces"].cpu().numpy()
            del blas
        except Exception as e:
            print(e, item["mesh_path"])
            octree = None
            points = None
            pyramid = None
            prefix = None
            max_level = 0
            vertices = None
            faces = None

        item = dict(
            **item,
            octree=octree,
            octree_points=points,
            octree_pyramid=pyramid,
            octree_prefix=prefix,
            octree_max_level=max_level,
            mesh_vertices=vertices,
            mesh_faces=faces,
        )

        # if not load_from_cache:
        #     with open(cache_path, "wb") as f:
        #         pickle.dump(item, f)

        return item

    ds = ray.data.from_items(items, parallelism=128)
    ds = ds.map(to_octree, compute=ActorPoolStrategy(4, 6), num_gpus=0.5)
    ds.write_parquet("data/ShapeNetCoreV2/octrees2.parquet")


if __name__ == "__main__":
    convert_to_octrees(
        Path("data/ShapeNetCore.v2"),
        ["car", "chair", "airplane"],
    )
