"""Lightning utilities."""
import functools
import logging
import warnings
from contextlib import contextmanager
from typing import Callable, Iterable, TypeVar

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

BatchType = TypeVar("BatchType")


class Lightning(pl.LightningModule):
    """Light-weight wrapper around a model and loss for supervised training."""

    # pylint:disable=too-many-ancestors,arguments-differ

    def __init__(
        self, model: nn.Module, loss: Callable[[BatchType], Tensor], config: dict
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.hparams.update(config)

    def configure_optimizers(self) -> Optimizer:
        return Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch: BatchType, _: int) -> Tensor:
        loss = self.loss_fn(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: BatchType, _: int) -> Tensor:
        loss = self.loss_fn(batch)
        self.log("val/loss", loss)
        return loss


def train_lite(
    model: Lightning,
    datamodule: pl.LightningDataModule,
    config: dict,
    callbacks: Iterable[pl.Callback] = (),
) -> dict:
    """Optimizes a model with minimal configurations.

    Args:
        model: The lightning model
        datamodule: The dataset module
        config: Dictionary with training configurations
        callbacks: Pytorch Lightning callbacks to pass to the trainer

    Returns:
        Dictionary with staticts of the trained model on validation data
    """
    # noinspection PyTypeChecker
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        logger=False,
        callbacks=[
            pl.callbacks.EarlyStopping("val/loss", check_on_train_epoch_end=False)
        ]
        + list(callbacks),
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0,  # don't show progress bar for model training
        weights_summary=None,  # don't print summary before training
        checkpoint_callback=False,  # don't save last model checkpoint
    )
    trainer.fit(model, datamodule=datamodule)
    epochs = trainer.current_epoch + 1
    # Assume there is only one validation dataloader
    (info,) = trainer.validate(model, datamodule=datamodule, verbose=False)
    info.update(epochs=epochs)
    return info


def suppress_info_logging():
    """Silences messages related to GPU/TPU availability."""
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


@contextmanager
def suppress_datamodule_warnings():
    """Ignores the DataModule.prepare_data deprecation warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            module="pytorch_lightning.core.datamodule",
            message="DataModule.prepare_data has already been called",
        )
        yield


@contextmanager
def suppress_dataloader_warnings(num_workers: bool = True, shuffle: bool = False):
    """Ignore PyTorch Lightning warnings regarding dataloaders.

    Args:
        num_workers: include number-of-workers warnings
        shuffle: include val/test dataloader shuffling warnings
    """
    suppress = functools.partial(
        warnings.filterwarnings,
        "ignore",
        module="pytorch_lightning.trainer.data_loading",
    )
    with warnings.catch_warnings():
        if num_workers:
            suppress(message=".*Consider increasing the value of the `num_workers`.*")
        if shuffle:
            suppress(message="Your .+_dataloader has `shuffle=True`")
        yield
