# pylint:disable=missing-docstring
import pytorch_lightning as pl

import lqsvg.torch.named as nt

from data import build_datamodule  # pylint:disable=wrong-import-order
from models import LightningModel  # pylint:disable=wrong-import-order
from policy import make_worker  # pylint:disable=wrong-import-order
from utils import suppress_dataloader_warning  # pylint:disable=wrong-import-order


@nt.suppress_named_tensor_warning()
def main():
    env_config = dict(n_state=2, n_ctrl=2, horizon=100, num_envs=100)
    worker = make_worker(env_config)
    model = LightningModel(worker.get_policy(), worker.env)
    datamodule = build_datamodule(worker, total_trajs=100)

    logger = pl.loggers.WandbLogger(
        name="SVG Prediction",
        offline=True,
        project="LQG-SVG",
        log_model=False,
        entity="angelovtt",
        tags="v0",
    )
    trainer = pl.Trainer(logger=logger, max_epochs=20)
    with suppress_dataloader_warning():
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
