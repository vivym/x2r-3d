from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl

from x2r_3d.data.datasets.sdf import build_sdf_dataset, build_sdf_train_iter, build_sdf_val_iter


class SDFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sdf_path: Union[str, Path],
        batch_size: int,
        num_workers: int = 64,
        random_seed: Optional[int] = None,
        prefetch_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.sdf_path = sdf_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.prefetch_blocks = prefetch_blocks

    def setup(self, stage: Optional[str] = None):
        self.ds = build_sdf_dataset(
            sdf_path=self.sdf_path,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return build_sdf_train_iter(
            self.ds,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            prefetch_blocks=self.prefetch_blocks,
        )

    def val_dataloader(self):
        return build_sdf_val_iter(
            self.ds,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            prefetch_blocks=self.prefetch_blocks,
        )
