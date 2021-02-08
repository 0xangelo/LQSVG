# pylint:disable=missing-docstring
import pytorch_lightning as pl

from data import build_datamodule
from models import LightningModel
from policy import make_worker


def main():
    worker = make_worker(num_envs=100)
    model = LightningModel(worker.get_policy(), worker.env)
    datamodule = build_datamodule(worker, total_trajs=100)

    logger = pl.loggers.WandbLogger(
        name="SVG Prediction",
        offline=False,
        project="LQG-SVG",
        log_model=False,
        entity="angelovtt",
        tags="v0",
    )
    trainer = pl.Trainer(logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
