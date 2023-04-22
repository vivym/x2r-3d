from pathlib import Path
from typing import Optional, List

import ray
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from ray.util import ActorPool
from tqdm import tqdm


@ray.remote(num_cpus=2, num_gpus=0.2)
class Worker:
    def train(self, object_id: str, tags: Optional[List[str]] = None, ckpt_path: Optional[Path] = None):
        from x2r_3d.accelstructs.octree_as import OctreeAS
        from x2r_3d.data.datamodules.sdf import SDFDataModule
        from x2r_3d.models.sdf.neural_sdf import NeuralSDF
        from x2r_3d.models.grids.octree_grid import OctreeGrid
        from x2r_3d.utils.logger import WandbLogger

        if tags is None:
            tags = []

        pl.seed_everything(2333)

        root_path = Path("data/ShapeNetCoreV2")
        sdf_path = root_path / "sdf" / (object_id + ".safetensors")
        octree_path = root_path / "octrees" / (object_id + ".pkl")

        dm = SDFDataModule(
            sdf_path=sdf_path,
            batch_size=512,
            num_workers=16,
            random_seed=2333,
        )

        model = NeuralSDF(
            grid=OctreeGrid(
                accelstruct=OctreeAS(octree_path),
                feature_dim=16,
                base_lod=2,
                num_lods=5,
                interpolation_type="linear",
                aggregation_type="sum",
                feature_std=0.01,
                feature_bias=0.,
            ),
        )

        trainer = pl.Trainer(
            devices=1,
            max_steps=30_000,
            val_check_interval=10_000,
            precision=16,
            logger=WandbLogger(
                project="x2r-3d",
                entity="viv",
                name="shapenet-nglod-batch-training",
                tags=[object_id] + tags,
            ),
            enable_progress_bar=False,
            callbacks=[
                ModelCheckpoint(
                    filename="ckpt_{step:08d}",
                    auto_insert_metric_name=False,
                    save_last=True,
                    save_top_k=-1,
                    every_n_train_steps=5_000
                ),
                LearningRateMonitor(),
            ],
            default_root_dir="./wandb",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule=dm)


def main():
    root_path = Path("data/ShapeNetCoreV2")
    object_ids = [p.stem for p in (root_path / "sdf").glob("*.safetensors")]

    pool = ActorPool([Worker.remote() for _ in range(15)])

    results = pool.map_unordered(
        lambda worker, object_id: worker.train.remote(object_id),
        object_ids,
    )

    for _ in tqdm(results):
        ...


if __name__ == "__main__":
    main()
