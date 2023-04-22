from pathlib import Path
from typing import Optional, Union

import pandas as pd
import ray.data
from safetensors.numpy import load_file


def build_sdf_dataset(
    sdf_path: Union[str, Path],
    num_workers: int = 64,
) -> ray.data.Dataset:
    sdf_dict = load_file(sdf_path)
    coords, sdf = sdf_dict["coords"], sdf_dict["sdf"]

    df = pd.DataFrame({"coords": [x for x in coords], "sdf": sdf})
    ds = ray.data.from_pandas(df)
    return ds.repartition(num_workers)


def build_sdf_train_iter(
    ds: ray.data.Dataset,
    batch_size: int,
    random_seed: Optional[int] = None,
    prefetch_blocks: int = 0,
) -> ray.data.DatasetIterator:
    pipe = ds.repeat()
    pipe = pipe.random_shuffle_each_window(seed=random_seed)

    return pipe.iter_torch_batches(prefetch_blocks=prefetch_blocks, batch_size=batch_size)


class IterDataset:
    def __init__(
        self,
        pipe: ray.data.DatasetPipeline,
        batch_size: int,
        prefetch_blocks: int = 0,
    ) -> None:
        self.epoch_pipes = pipe.iter_epochs()
        self.batch_size = batch_size
        self.prefetch_blocks = prefetch_blocks

    def __iter__(self):
        epoch_pipe = next(self.epoch_pipes)

        yield from epoch_pipe.iter_torch_batches(
            prefetch_blocks=self.prefetch_blocks, batch_size=self.batch_size
        )


def build_sdf_val_iter(
    ds: ray.data.Dataset,
    batch_size: int,
    random_seed: Optional[int] = None,
    prefetch_blocks: int = 0,
) -> ray.data.DatasetIterator:
    ds = ds.random_shuffle(seed=random_seed)
    ds = ds.limit(500_000)
    ds = ds.repartition(32)

    pipe = ds.repeat()

    return IterDataset(
        pipe, batch_size=batch_size, prefetch_blocks=prefetch_blocks
    )
