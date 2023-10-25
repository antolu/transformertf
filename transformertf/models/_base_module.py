from __future__ import annotations

import functools
import typing

import lightning as L
import lightning.pytorch.utilities
import torch

from ..config import BaseConfig
from ..data import TimeSeriesSample
from ..utils import configure_lr_scheduler, configure_optimizers, ops

MODEL_INPUT = typing.Union[torch.Tensor, dict[str, torch.Tensor]]
MODEL_OUTPUT = typing.Union[torch.Tensor, dict[str, torch.Tensor]]
MODEL_STATES = typing.Union[torch.Tensor, dict[str, torch.Tensor]]

STEP_OUTPUT = typing.Union[MODEL_OUTPUT, dict[str, MODEL_OUTPUT]]
EPOCH_OUTPUT = list[STEP_OUTPUT]


if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="LightningModuleBase")
    from ..utils import OPTIMIZER_DICT


class LightningModuleBase(L.LightningModule):
    _lr_scheduler: str | typing.Type[
        torch.optim.lr_scheduler.LRScheduler
    ] | functools.partial | None

    def __init__(self) -> None:
        super().__init__()

        self._train_outputs: list[MODEL_OUTPUT] = []
        self._val_outputs: list[MODEL_OUTPUT] = []
        self._test_outputs: list[MODEL_OUTPUT] = []
        self._inference_outputs: list[MODEL_OUTPUT] = []

        self._lr_scheduler = None

    @classmethod
    def from_config(
        cls: typing.Type[SameType], config: BaseConfig, **kwargs: typing.Any
    ) -> SameType:
        return cls(**cls.parse_config_kwargs(config, **kwargs))

    @classmethod
    def parse_config_kwargs(
        cls, config: BaseConfig, **kwargs: typing.Any
    ) -> dict[str, typing.Any]:
        default_kwargs = dict(
            optimizer=config.optimizer,
            optimizer_kwargs=config.optimizer_kwargs,
            lr_scheduler=config.lr_scheduler,
            lr_scheduler_interval=config.lr_scheduler_interval,
            max_epochs=config.num_epochs,
            reduce_on_plateau_patience=config.patience,
            log_grad_norm=config.log_grad_norm,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            validate_every_n_epochs=config.validate_every,
        )

        default_kwargs.update(kwargs)

        return default_kwargs

    def on_train_start(self) -> None:
        self._train_outputs = []

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

        super().on_validation_epoch_start()

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._val_outputs.append(ops.to_cpu(ops.detach(outputs)))

    def on_test_epoch_start(self) -> None:
        self._test_outputs = []

        super().on_test_epoch_start()

    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._test_outputs.append(ops.to_cpu(ops.detach(outputs)))

    def on_predict_epoch_start(self) -> None:
        self._inference_outputs = []

        super().on_predict_epoch_start()

    def on_predict_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._inference_outputs.append(ops.to_cpu(ops.detach(outputs)))

    def common_log_step(
        self,
        loss: dict[str, torch.Tensor],
        stage: typing.Literal["train", "validation", "test", "inference"],
    ) -> dict[str, torch.Tensor]:
        log_dict = {k + f"/{stage}": v for k, v in loss.items() if k != "loss"}

        if self.logger is not None:
            self.log(
                f"loss/{stage}",
                loss["loss"],
                on_step=stage == "train",
                prog_bar=stage == "train",
            )
            self.log_dict(log_dict)

        return log_dict

    def on_before_optimizer_step(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        if "log_grad_norm" in self.hparams and self.hparams["log_grad_norm"]:
            self.log_dict(
                lightning.pytorch.utilities.grad_norm(self, norm_type=2)
            )

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OPTIMIZER_DICT:
        lr: float = self.hparams["lr"]
        if lr == "auto":
            lr = 1e-3
        optimizer = configure_optimizers(
            self.hparams["optimizer"],
            lr=lr,
            weight_decay=self.hparams.get("weight_decay", 0.0),
            momentum=self.hparams.get("momentum", 0.0),
            **self.hparams.get("optimizer_kwargs", {}),
        )(self.parameters())

        if self._lr_scheduler is None:
            return optimizer

        scheduler_dict = configure_lr_scheduler(
            optimizer,
            lr_scheduler=self._lr_scheduler,
            monitor="loss/validation",
            scheduler_interval=self.hparams["lr_scheduler_interval"],
            max_epochs=self.hparams["max_epochs"],
            reduce_on_plateau_patience=self.hparams[
                "reduce_on_plateau_patience"
            ],
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
