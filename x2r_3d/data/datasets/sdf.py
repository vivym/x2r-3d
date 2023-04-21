from pathlib import Path
from typing import Optional, Union

import pandas as pd
import ray.data
from safetensors.numpy import load_file


def build_sdf_dataset(
    sdf_path: Union[str, Path],
    batch_size: int,
    num_workers: int = 16,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
    prefetch_blocks: int = 0,
):
    sdf_dict = load_file(sdf_path)
    coords, sdf = sdf_dict["coords"], sdf_dict["sdf"]

    df = pd.DataFrame({"coords": [x for x in coords], "sdf": sdf})
    ds = ray.data.from_pandas(df)
    ds = ds.repartition(num_workers)
    pipe = ds.repeat()

    if shuffle:
        pipe = pipe.random_shuffle_each_window(seed=random_seed)

    return pipe.iter_torch_batches(prefetch_blocks=prefetch_blocks, batch_size=batch_size)
