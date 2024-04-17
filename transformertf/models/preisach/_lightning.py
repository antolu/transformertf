from __future__ import annotations

import functools
import typing

import torch

from ...data import TimeSeriesSample
from ...hysteresis.base import BaseHysteresis
from .._base_module import LR_CALL_TYPE, OPT_CALL_TYPE, LightningModuleBase
from ._config import PreisachConfig

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="PreisachModule")


class PreisachModule(LightningModuleBase):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        optimizer: str | OPT_CALL_TYPE = "adam",
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        reduce_on_plateau_patience: int = 50,
        lr_scheduler: str | LR_CALL_TYPE | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
        max_epochs: int = 1000,
        criterion: torch.nn.Module | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        """
        This module implements a PyTorch Lightning module for the
        Differentiable Preisach Model.
        """
        criterion = criterion or torch.nn.MSELoss()
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs or {},
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            max_epochs=max_epochs,
            log_grad_norm=log_grad_norm,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
            criterion=criterion,
        )
        super().save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self.model = BaseHysteresis()

    @classmethod
    def from_config(  # type: ignore[override]
        cls: type[SameType],
        config: PreisachConfig,
        criterion: torch.nn.Module | None = None,
        lr_scheduler: (
            str | type[torch.optim.lr_scheduler.LRScheduler] | functools.partial | None
        ) = None,
        **kwargs: typing.Any,
    ) -> SameType:
        default_kwargs = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "momentum": config.momentum,
            "optimizer": config.optimizer,
            "optimizer_kwargs": config.optimizer_kwargs,
            "reduce_on_plateau_patience": config.patience,
            "lr_scheduler_interval": config.lr_scheduler_interval,
            "max_epochs": config.num_epochs,
            "log_grad_norm": config.log_grad_norm,
            "criterion": criterion or torch.nn.MSELoss(),
            "lr_scheduler": lr_scheduler or config.lr_scheduler,
        }
        kwargs.update(default_kwargs)

        return cls(**kwargs)

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: PreisachConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        # no need for custom implementation
        return super().parse_config_kwargs(config, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.model.mode = 0

        return super().on_train_epoch_start()

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]
        input_ = batch["input"]

        if input_.ndim > 1:
            input_ = input_.squeeze()
            target = target.squeeze()

        model_output = self.model(input_)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        self.common_log_step({"loss": loss}, "train")

        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        self.model.mode = 1

        return super().on_validation_epoch_start()

    def validation_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]
        input_ = batch["input"]

        if input_.ndim > 1:
            input_ = input_.squeeze()
            target = target.squeeze()

        model_output = self.model(input_)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        self.common_log_step({"loss": loss}, "validation")

        return {"loss": loss}
