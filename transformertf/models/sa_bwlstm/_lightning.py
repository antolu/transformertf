from __future__ import annotations

import collections.abc
import functools
import itertools
import typing

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ...data import TimeSeriesSample
from ...utils import configure_lr_scheduler, configure_optimizers
from .._base_module import DEFAULT_LOGGING_METRICS, MetricLiteral
from ..bwlstm import BWLSTM3, BoucWenLoss


class SABWLSTM(BWLSTM3):
    def __init__(
        self,
        n_features: int = 1,
        num_layers: int | tuple[int, int, int] = 3,
        d_model: int | tuple[int, int, int] = 350,
        n_dim_fc: int | tuple[int, int, int] | None = None,
        dropout: float | tuple[float, float, float] = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        optimizer: str
        | functools.partial
        | typing.Callable[
            [typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ] = "adam",
        lr: float | typing.Literal["auto"] = 1e-3,
        weight_decay: float | None = None,
        momentum: float | None = None,
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        lr_scheduler: str
        | type[torch.optim.lr_scheduler.LRScheduler]
        | functools.partial
        | typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        | None = None,
        monitor: str | None = None,
        scheduler_interval: typing.Literal["step", "epoch"] = "epoch",
        max_epochs: int | None = None,
        reduce_on_plateau_patience: int | None = None,
        lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
        sa_optimizer: str
        | functools.partial
        | typing.Callable[
            [typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ] = "adam",
        sa_lr: float | typing.Literal["auto"] = 1e-3,
        sa_weight_decay: float | None = None,
        sa_momentum: float | None = None,
        sa_optimizer_kwargs: dict[str, typing.Any] | None = None,
        sa_lr_scheduler: str
        | type[torch.optim.lr_scheduler.LRScheduler]
        | functools.partial
        | typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        | None = None,
        sa_monitor: str | None = None,
        sa_scheduler_interval: typing.Literal["step", "epoch"] = "epoch",
        sa_max_epochs: int | None = None,
        sa_reduce_on_plateau_patience: int | None = None,
        sa_lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
        lbfgs_start: typing.Literal[False] | int = False,
        lbfgs_lr: float = 1.0,
        lbfgs_max_iter: int = 20,
        lbfgs_history_size: int = 5,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__(
            n_features=n_features,
            num_layers=num_layers,
            d_model=d_model,
            n_dim_fc=n_dim_fc,
            dropout=dropout,
            loss_weights=loss_weights,
            log_grad_norm=log_grad_norm,
            compile_model=compile_model,
            logging_metrics=logging_metrics,
        )
        self.save_hyperparameters()

        if optimizer_kwargs is None:
            self.hparams["optimizer_kwargs"] = {}
        if lr_scheduler_kwargs is None:
            self.hparams["lr_scheduler_kwargs"] = {}
        if sa_optimizer_kwargs is None:
            self.hparams["sa_optimizer_kwargs"] = {}
        if sa_lr_scheduler_kwargs is None:
            self.hparams["sa_lr_scheduler_kwargs"] = {}

        self.criterion.trainable = True

        self.automatic_optimization = False

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        if (
            self.hparams["lbfgs_start"]
            and self.current_epoch >= self.hparams["lbfgs_start"]
        ):
            return self.second_order_step(batch, batch_idx)

        return self.first_order_step(batch, batch_idx)

    def first_order_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        try:
            target = batch["target"]
        except KeyError as e:
            msg = (
                "The batch must contain a target key. "
                "This is probably due to using a dataset without targets "
                "(e.g. test or predict)."
            )
            raise ValueError(msg) from e

        output = self(batch["input"])

        loss, losses = self.criterion(output, target, return_all=True)

        for optimizer in self.optimizers()[:2]:
            optimizer.zero_grad()

        self.manual_backward(loss)

        self.optimizers()[0].step()

        self.criterion.invert_gradients()
        self.optimizers()[1].step()

        loss_weights = {
            "loss_weight/alpha": self.criterion.alpha.item(),
            "loss_weight/beta": self.criterion.beta.item(),
            "loss_weight/gamma": self.criterion.gamma.item(),
            "loss_weight/kappa": self.criterion.kappa.item(),
            "loss_weight/eta": self.criterion.eta.item(),
        }

        losses = self.rename_losses_dict(losses)
        self.common_log_step(losses | loss_weights, "train")

        return losses | {"output": output}

    def second_order_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        try:
            target = batch["target"]
        except KeyError as e:
            msg = (
                "The batch must contain a target key. "
                "This is probably due to using a dataset without targets "
                "(e.g. test or predict)."
            )
            raise ValueError(msg) from e

        def closure() -> torch.Tensor:
            self.optimizers()[2].zero_grad()

            output = self(batch["input"])
            loss = self.criterion(output, target)
            self.manual_backward(loss)

            return loss

        output = self(batch["input"])
        _, losses = self.criterion(output, target, return_all=True)
        losses = self.rename_losses_dict(losses)
        self.common_log_step(losses, "train")

        self.optimizers()[2].step(closure)

        loss_weights = {
            "loss_weight/alpha": self.criterion.alpha.item(),
            "loss_weight/beta": self.criterion.beta.item(),
            "loss_weight/gamma": self.criterion.gamma.item(),
            "loss_weight/kappa": self.criterion.kappa.item(),
            "loss_weight/eta": self.criterion.eta.item(),
        }

        return losses | {"output": output} | loss_weights

    @staticmethod
    def rename_losses_dict(losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            f"loss_component/{k}" if k != "loss" else k: v for k, v in losses.items()
        }

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = configure_optimizers(
            self.hparams["optimizer"],
            lr=self.hparams["lr"],
            weight_decay=self.hparams.get("weight_decay", None),
            momentum=self.hparams.get("momentum", None),
            **self.hparams.get("optimizer_kwargs", {}),
        )(
            itertools.chain(
                self.bwlstm1.parameters(),
                self.bwlstm2.parameters(),
                self.bwlstm3.parameters(),
            )
        )

        sa_optimizer = configure_optimizers(
            self.hparams["sa_optimizer"],
            lr=self.hparams["sa_lr"],
            weight_decay=self.hparams.get("sa_weight_decay", None),
            momentum=self.hparams.get("sa_momentum", None),
            **self.hparams.get("sa_optimizer_kwargs", {}),
        )(self.criterion.parameters())

        if self.hparams.get("lr_scheduler"):
            lr_scheduler = configure_lr_scheduler(
                optimizer,
                self.hparams["lr_scheduler"],
                monitor=self.hparams.get("monitor", None),
                scheduler_interval=self.hparams.get("scheduler_interval", None),
                max_epochs=self.hparams.get("max_epochs", None),
            )
        else:
            lr_scheduler = None

        if lr_scheduler is not None:
            opt1 = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            opt1 = optimizer

        if self.hparams.get("sa_lr_scheduler"):
            sa_lr_scheduler = configure_lr_scheduler(
                sa_optimizer,
                self.hparams["sa_lr_scheduler"],
                monitor=self.hparams.get("sa_monitor", None),
                scheduler_interval=self.hparams.get("sa_scheduler_interval", None),
                max_epochs=self.hparams.get("sa_max_epochs", None),
            )
        else:
            sa_lr_scheduler = None

        if sa_lr_scheduler is not None:
            opt2 = {"optimizer": sa_optimizer, "lr_scheduler": sa_lr_scheduler}
        else:
            opt2 = sa_optimizer

        optimizers = [opt1, opt2]

        if self.hparams.get("lbfgs_start"):
            optimizers.append(
                torch.optim.LBFGS(
                    itertools.chain(
                        self.bwlstm1.parameters(),
                        self.bwlstm2.parameters(),
                        self.bwlstm3.parameters(),
                    ),
                    lr=self.hparams["lbfgs_lr"],
                    max_iter=self.hparams["lbfgs_max_iter"],
                    history_size=self.hparams["lbfgs_history_size"],
                )
            )

        return optimizers
