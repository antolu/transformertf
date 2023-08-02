from __future__ import annotations

import functools
import inspect
import typing

import lightning as L
import lightning.pytorch.utilities
import pytorch_optimizer as py_optim
import torch

from ..config import BaseConfig
from ..data import TimeSeriesSample
from ..utils import ops

MODEL_INPUT = typing.Union[torch.Tensor, dict[str, torch.Tensor]]
MODEL_OUTPUT = typing.Union[torch.Tensor, dict[str, torch.Tensor]]
MODEL_STATES = typing.Union[torch.Tensor, dict[str, torch.Tensor]]

STEP_OUTPUT = typing.Union[MODEL_OUTPUT, dict[str, MODEL_OUTPUT]]
EPOCH_OUTPUT = list[STEP_OUTPUT]

LR_SCHEDULER_DICT = typing.TypedDict(
    "LR_SCHEDULER_DICT",
    {
        "scheduler": typing.Union[
            torch.optim.lr_scheduler.LRScheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ],
        "monitor": str,
        "interval": typing.Literal["epoch", "step"],
    },
)
OPTIMIZER_DICT = typing.TypedDict(
    "OPTIMIZER_DICT",
    {
        "optimizer": torch.optim.Optimizer,
        "lr_scheduler": typing.Union[
            typing.Union[
                torch.optim.lr_scheduler.LRScheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ],
            None,
            LR_SCHEDULER_DICT,
        ],
    },
)

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="LightningModuleBase")


class LightningModuleBase(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self._train_outputs: list[MODEL_OUTPUT] = []
        self._val_outputs: list[MODEL_OUTPUT] = []
        self._test_outputs: list[MODEL_OUTPUT] = []
        self._inference_outputs: list[MODEL_OUTPUT] = []

    @classmethod
    def from_config(
        cls: typing.Type[SameType], config: BaseConfig
    ) -> SameType:
        raise NotImplementedError

    def on_train_start(self) -> None:
        self._train_outputs = []

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

        super().on_validation_start()

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

        super().on_test_start()

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

        super().on_predict_start()

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

        optimizer: torch.optim.Optimizer
        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=self.hparams["weight_decay"],
                **self.hparams["optimizer_kwargs"],
            )
        elif self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=self.hparams["weight_decay"],
                **self.hparams["optimizer_kwargs"],
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams["weight_decay"],
                **self.hparams["optimizer_kwargs"],
            )
        elif self.hparams["optimizer"] == "ranger":
            optimizer = py_optim.Ranger(
                self.parameters(),
                lr=lr,
                weight_decay=self.hparams["weight_decay"],
                **self.hparams["optimizer_kwargs"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams['optimizer']}")

        if self._lr_scheduler is not None:
            if isinstance(self._lr_scheduler, str):
                if self._lr_scheduler == "plateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=optimizer,
                        mode="min",
                        factor=0.1,
                        verbose=True,
                        patience=self.hparams["reduce_on_plateau_patience"],
                    )
                if self._lr_scheduler == "constant_then_cosine":
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        [  # type: ignore
                            torch.optim.lr_scheduler.ConstantLR(
                                factor=1.0,
                                optimizer=optimizer,
                                total_iters=int(
                                    0.75 * self.hparams["max_epochs"]
                                ),
                            ),
                            torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer=optimizer,
                                T_max=int(0.25 * self.hparams["max_epochs"]),
                            ),
                        ],
                        milestones=[int(0.75 * self.hparams["max_epochs"])],
                    )
                else:
                    raise ValueError(
                        f"Unknown lr_scheduler: {self._lr_scheduler}"
                    )
            elif isinstance(self._lr_scheduler, functools.partial):
                scheduler = self._lr_scheduler(optimizer=optimizer)
            elif inspect.isclass(self._lr_scheduler):
                scheduler = self._lr_scheduler(optimizer=optimizer)
            else:
                raise NotImplementedError(
                    "Learning rate schedulers are not implemented yet."
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/validation",
                    "interval": self.hparams["lr_scheduler_interval"],
                },
            }

        return optimizer
