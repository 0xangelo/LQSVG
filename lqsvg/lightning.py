"""Lightning utilities."""
import functools
import logging
import warnings
from contextlib import contextmanager
from typing import Callable, TypeVar

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
        self.loss = loss
        self.hparams.update(config)

    def configure_optimizers(self) -> Optimizer:
        return Adam(
            self.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch: BatchType) -> Tensor:
        loss = self.loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: BatchType) -> Tensor:
        loss = self.loss(batch)
        self.log("val/loss", loss)
        return loss


def suppress_lightning_info_logging():
    """Silences messages related to GPU/TPU availability."""
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
    logging.getLogger("lightning").setLevel(logging.WARNING)


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