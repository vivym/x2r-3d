import multiprocessing as mp
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
from tqdm import tqdm

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


def make_mesh_watertight(obj_path: Path):
    mesh_path = obj_path / "models" / "model_normalized.obj"
    output_path = obj_path / "models" / "model_normalized_watertight.obj"

    try:
        v, f = pcu.load_mesh_vf(str(mesh_path))
        vm, fm = pcu.make_mesh_watertight(v, f, resolution=20_000, seed=2333)
        nm = pcu.estimate_mesh_vertex_normals(vm, fm)

        pcu.save_mesh_vfn(str(output_path), vm, fm, nm)
    except Exception as e:
        print(e)


def main():
    root_path = Path("data/ShapeNetCore.v2")
    # categories = ["car", "chair", "airplane"]
    categories = ["airplane"]

    with mp.Pool(processes=16) as pool:
        for category in categories:
            synset_id = category_to_synset_id[category]
            category_root_path = root_path / synset_id
            obj_paths = sorted(category_root_path.glob("*"))[2000:]

            results = tqdm(
                pool.imap_unordered(make_mesh_watertight, obj_paths),
                total=len(obj_paths),
            )
            for _ in results:
                ...


if __name__ == "__main__":
    main()
