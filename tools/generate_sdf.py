import pickle
import multiprocessing as mp
from pathlib import Path

import ray
import numpy as np
import point_cloud_utils as pcu
import zstandard as zstd
from tqdm import tqdm
from safetensors.numpy import save_file

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


@ray.remote(num_cpus=12)
def compute_sdf(pts_path: Path):
    obj_id = pts_path.stem
    octree_path = pts_path.parent.parent / "octrees.cache" / obj_id
    sdf_path = pts_path.parent.parent / "sdf" / (obj_id + ".safetensors")

    if sdf_path.exists():
        return

    with open(octree_path, "rb") as f:
        octree_data = pickle.load(f)

    with open(pts_path, "rb") as f:
        pts = pickle.load(f)

    vm = octree_data["mesh_vertices"]
    fm = octree_data["mesh_faces"]

    sdf, _, _ = pcu.signed_distance_to_mesh(pts, vm, fm)
    save_file(dict(coords=pts, sdf=sdf), sdf_path)


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids, num_returns=1)
        yield ray.get(done[0])


def main():
    root_path = Path("data/ShapeNetCoreV2")

    results = [
        compute_sdf.remote(pts_path)
        for pts_path in sorted((root_path / "pts").glob("*"))
    ]
    print(len(results))

    for _ in tqdm(to_iterator(results), total=len(results)):
        ...


if __name__ == "__main__":
    main()
