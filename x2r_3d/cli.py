import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback


def main():
    LightningCLI(
        pl.LightningModule, pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    main()
