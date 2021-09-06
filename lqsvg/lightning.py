"""Lightning utilities."""
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
