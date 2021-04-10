"""
# Value gradient error for linear policies in LQG

Experiment description on [Overleaf](https://www.overleaf.com/read/cmbgmxxpxqzr).

**Versioning:** [CalVer](https://calver.org) `MM.DD.MICRO`
"""
from __future__ import annotations

import logging
import os
import os.path as osp

import pytorch_lightning as pl
import ray
import wandb
from ray import tune
from torch import Tensor

import lqsvg
import lqsvg.envs.lqr.utils as lqg_util
import lqsvg.experiment.utils as utils
import lqsvg.torch.named as nt
from lqsvg.experiment.data import build_datamodule
from lqsvg.experiment.models import LightningModel, RecurrentModel
from lqsvg.experiment.worker import make_worker


class InputStatistics(pl.callbacks.Callback):
    # pylint:disable=missing-class-docstring,too-many-arguments
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        del trainer, outputs, batch_idx, dataloader_idx
        obs, act, new_obs = batch
        pl_module.log("train/obs-mean", obs.mean())
        pl_module.log("train/obs-std", obs.std())
        pl_module.log("train/act-mean", act.mean())
        pl_module.log("train/act-std", act.std())
        pl_module.log("train/new_obs-mean", new_obs.mean())
        pl_module.log("train/new_obs-std", new_obs.std())


# noinspection PyAttributeOutsideInit,PyAbstractClass
class Experiment(tune.Trainable):
    # pylint:disable=missing-class-docstring,missing-function-docstring,abstract-method,attribute-defined-outside-init
    def setup(self, config: dict):
        os.environ["WANDB_SILENT"] = "true"
        cwd = config.pop("wandb_dir")
        tags = config.pop("wandb_tags", [])
        self.run = wandb.init(
            dir=cwd,
            name="SVG Prediction",
            config=config,
            project="LQG-SVG",
            entity="angelovtt",
            tags=[utils.calver()] + tags,
            reinit=True,
            mode="online",
            save_code=True,
        )

        self.make_worker()
        self.make_model()
        self.make_datamodule()
        self.make_lightning_trainer()
        #         self.make_artifact()
        utils.suppress_lightning_info_logging()

    @property
    def hparams(self):
        return self.run.config

    def make_worker(self):
        with nt.suppress_named_tensor_warning():
            self.worker = make_worker(
                env_config=self.hparams.env_config,
                # TorchPolicy internally pulls the 'policy' key from config
                policy_config=dict(self.hparams),
                log_level=logging.WARNING,
            )

    def make_model(self):
        cls = RecurrentModel if self.hparams.recurrent_training else LightningModel
        self.model = cls(self.worker.get_policy(), self.worker.env)
        self.model.hparams.learning_rate = self.hparams.learning_rate
        self.model.hparams.mc_samples = self.hparams.mc_samples
        self.model.hparams.weight_decay = self.hparams.weight_decay
        self.model.hparams.empvar_samples = self.hparams.empvar_samples

    def make_datamodule(self):
        self.datamodule = build_datamodule(
            self.worker, total_trajs=self.hparams.total_trajs
        )

    def make_lightning_trainer(self):
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )

        early_stopping = pl.callbacks.EarlyStopping(
            monitor=LightningModel.early_stop_on,
            min_delta=float(self.hparams.improvement_delta),
            patience=int(self.hparams.patience),
            mode="min",
            strict=True,
        )
        callbacks = [early_stopping, InputStatistics()]
        # callbacks += [self._checkpoint_callback()]
        self.trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            num_sanity_val_steps=2,
            callbacks=callbacks,
            max_epochs=self.hparams.max_epochs,
            progress_bar_refresh_rate=0,  # don't show progress bar for model training
            weights_summary=None,  # don't print summary before training
        )

    def _checkpoint_callback(self) -> pl.callbacks.ModelCheckpoint:
        return pl.callbacks.ModelCheckpoint(
            dirpath=osp.join(self.run.dir, "checkpoints"),
            monitor=LightningModel.early_stop_on,
            save_top_k=-1,
            period=10,
            save_last=True,
        )

    def make_artifact(self):
        env = self.worker.env
        self.artifact = wandb.Artifact(
            f"svg_prediction-lqg{env.n_state}.{env.n_ctrl}.{env.horizon}", type="model"
        )

    def step(self) -> dict:
        with self.run:
            self.log_env_info()
            self.datamodule.collect_trajectories(prog=False)
            with utils.suppress_dataloader_warning():
                self.trainer.fit(self.model, datamodule=self.datamodule)

                results = self.trainer.test(self.model, datamodule=self.datamodule)[0]
                self.run.summary.update(results)
            # self._try_save_artifact()

        return {tune.result.DONE: True, **results}

    def log_env_info(self):
        dynamics = self.worker.env.dynamics
        eigvals = lqg_util.stationary_eigvals(dynamics)
        tests = {
            "stability": lqg_util.isstable(eigvals=eigvals),
            "controllability": lqg_util.iscontrollable(dynamics),
        }
        self.run.summary.update(tests)
        self.run.summary.update({"passive_eigvals": wandb.Histogram(eigvals)})

    def _try_save_artifact(self):
        try:
            self.artifact.add_dir(self.trainer.checkpoint_callback.dirpath)
            self.run.log_artifact(self.artifact)
        except ValueError:
            # Sometimes add_dir fails with 'not a directory name'. Shall investigate
            pass

    def cleanup(self):
        self.run.finish()


def main():
    """Run hyperparameter sweep."""
    ray.init(logging_level=logging.WARNING)
    lqsvg.register_all()
    utils.suppress_lightning_info_logging()

    config = {
        "wandb_dir": os.getcwd(),
        "wandb_tags": "easy unstable controllable".split(),
        "env_config": dict(
            n_state=tune.randint(2, 11),
            n_ctrl=tune.randint(2, 11),
            horizon=tune.randint(1, 200),
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            controllable=True,
            transition_bias=False,
            rand_trans_cov=False,
            rand_init_cov=False,
            cost_linear=False,
            cost_cross=False,
            num_envs=100,
        ),
        "policy": {
            "module": {
                "policy_initializer": {"min_abs_eigv": 0.0, "max_abs_eigv": 1.0},
                "model_initializer": "xavier_uniform",
                "stationary_model": True,
                "residual_model": True,
                "model_input_norm": tune.grid_search(["LayerNorm", "BatchNorm"]),
            }
        },
        "recurrent_training": False,
        "learning_rate": tune.loguniform(1e-3, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "mc_samples": 32,
        "empvar_samples": 10,
        "total_trajs": 1000,
        "improvement_delta": 0.0,
        "patience": 3,
        "max_epochs": 1000,
    }

    tune.run(Experiment, config=config, num_samples=128, local_dir="./results")
    ray.shutdown()


if __name__ == "__main__":
    main()
