from __future__ import annotations

import typing

import lightning as L
import lightning.pytorch.utilities
import torch

from ..config import BaseConfig
from ..data import TimeSeriesSample
from ..nn import functional as F
from ..utils import configure_lr_scheduler, configure_optimizers, ops

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="LightningModuleBase")
    from ..utils import OPTIMIZER_DICT
    from .typing import LR_CALL_TYPE, MODEL_OUTPUT, OPT_CALL_TYPE, STEP_OUTPUT


class LightningModuleBase(L.LightningModule):
    _lr_scheduler: (
        str
        | typing.Callable[
            [tuple[typing.Any, ...]], torch.optim.lr_scheduler.LRScheduler
        ]
        | None
    )

    def __init__(
        self,
        optimizer: str | OPT_CALL_TYPE,
        optimizer_kwargs: dict[str, typing.Any],
        lr_scheduler: str | LR_CALL_TYPE | None,
        lr_scheduler_interval: str,
        max_epochs: int,
        reduce_on_plateau_patience: int,
        log_grad_norm: bool,
        lr: float,
        weight_decay: float,
        momentum: float,
        validate_every_n_epochs: int,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self._train_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._val_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._test_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._inference_outputs: dict[int, list[MODEL_OUTPUT]] = {}

        self._lr_scheduler = lr_scheduler

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
        self._train_outputs = {}

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = {}

        super().on_validation_epoch_start()

    def on_train_batch_end(
        self,
        outputs: torch.Tensor | typing.Mapping[str, typing.Any] | None,
        batch: typing.Any,
        batch_idx: int,
    ) -> None:

        if "prediction_type" not in self.hparams or (
            "prediction_type" in self.hparams
            and self.hparams["prediction_type"] == "point"
        ):
            assert outputs is not None
            assert "target" in batch
            assert isinstance(outputs, dict)
            other_metrics = self.calc_other_metrics(outputs, batch["target"])
            self.common_log_step(other_metrics, "train")

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._val_outputs:
            self._val_outputs[dataloader_idx] = []
        self._val_outputs[dataloader_idx].append(
            ops.to_cpu(ops.detach(outputs))
        )

        if "prediction_type" not in self.hparams or (
            "prediction_type" in self.hparams
            and self.hparams["prediction_type"] == "point"
        ):
            assert "target" in batch
            other_metrics = self.calc_other_metrics(outputs, batch["target"])
            self.common_log_step(other_metrics, "validation")

    def calc_other_metrics(
        self, outputs: STEP_OUTPUT, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Calculate other metrics from the outputs of the model.

        Parameters
        ----------
        outputs : STEP_OUTPUT
            The outputs of the model. Should contain keys "output" and "loss", and optionally "point_prediction".
            If the "point_prediction" key is present, it should be a tensor of shape (batch_size, prediction_horizon, 1),
            otherwise it will be calculated from the "output" tensor.
        target : torch.Tensor
            The target tensor. Should be a tensor of shape (batch_size, prediction_horizon, 1).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the calculated metrics.
        """
        loss_dict: dict[str, torch.Tensor] = {}

        if target.ndim == 3:
            target = target.squeeze(-1)

        assert isinstance(outputs, dict)
        if "point_prediction" in outputs:
            prediction = outputs["point_prediction"]
        else:
            prediction = outputs["output"]

        assert isinstance(prediction, torch.Tensor)
        if prediction.ndim == 3:
            prediction = prediction.squeeze(-1)

        loss_dict["MSE"] = torch.nn.functional.mse_loss(prediction, target)
        loss_dict["MAE"] = torch.nn.functional.l1_loss(prediction, target)
        loss_dict["MAPE"] = F.mape_loss(prediction, target)
        loss_dict["SMAPE"] = F.smape_loss(prediction, target)
        loss_dict["RMSE"] = torch.sqrt(loss_dict["MSE"])

        return loss_dict

    def on_test_epoch_start(self) -> None:
        self._test_outputs = {}

        super().on_test_epoch_start()

    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._test_outputs:
            self._test_outputs[dataloader_idx] = []
        self._test_outputs[dataloader_idx].append(
            ops.to_cpu(ops.detach(outputs))
        )

    def on_predict_epoch_start(self) -> None:
        self._inference_outputs = {}

        super().on_predict_epoch_start()

    def on_predict_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._inference_outputs:
            self._inference_outputs[dataloader_idx] = []
        self._inference_outputs[dataloader_idx].append(
            ops.to_cpu(ops.detach(outputs))
        )

    def common_log_step(
        self,
        loss: dict[str, torch.Tensor],
        stage: typing.Literal["train", "validation", "test", "inference"],
    ) -> dict[str, torch.Tensor]:
        log_dict = {k + f"/{stage}": v for k, v in loss.items()}

        if self.logger is not None:
            self.log_dict(
                log_dict,
                on_step=stage == "train",
                prog_bar=stage == "train",
                sync_dist=True,
            )

        return log_dict

    def on_before_optimizer_step(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        if "log_grad_norm" in self.hparams and self.hparams["log_grad_norm"]:
            self.log_dict(
                lightning.pytorch.utilities.grad_norm(self, norm_type=2)
            )

    def configure_optimizers(  # type: ignore[override]
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

    @property
    def validation_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        return self._val_outputs

    @property
    def test_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        return self._test_outputs

    @property
    def inference_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        return self._inference_outputs
