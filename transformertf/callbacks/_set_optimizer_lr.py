from __future__ import annotations

import logging
import os
import pathlib
import typing
from typing import Any

import lightning as L

from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class SetOptimizerLRCallback(L.pytorch.callbacks.Callback):
    """Sets learning rate based on contents of a file"""

    def __init__(
        self,
        lr_file: str | os.PathLike = "/tmp/lr.txt",
        on: typing.Literal["epoch", "step"] = "epoch",
        to: list[int] | None = None,
    ):
        super().__init__()
        self.lr_file = pathlib.Path(lr_file)
        self.on = on
        self.to = to

    def on_train_epoch_start(
        self, trainer: L.pytorch.Trainer, pl_module: LightningModuleBase
    ) -> None:
        if self.on == "epoch":
            self.set_lr(trainer)

    def on_train_batch_start(
        self,
        trainer: L.pytorch.Trainer,
        pl_module: LightningModuleBase,
        batch: typing.Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> None:
        if self.on == "step":
            self.set_lr(trainer)

    def set_lr(self, trainer: L.pytorch.Trainer) -> None:
        if not self.lr_file.exists():
            log.debug(f"File {self.lr_file} does not exist.")
            return

        with open(self.lr_file, encoding="utf-8") as f:
            try:
                contents = f.read().strip()
            except ValueError:
                log.exception(
                    f"Could not read LR from file {self.lr_file}, Contents: {contents}"
                )
                return

            try:
                lr = float(contents)
            except ValueError:
                log.exception(
                    f"Could not convert contents of file {self.lr_file} "
                    f"to float: {contents}"
                )
                return

        log.info(f"Setting LR to {lr}.")
        if self.to is None:
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
        else:
            for i in self.to:
                if i < len(trainer.optimizers):
                    for param_group in trainer.optimizers[i].param_groups:
                        param_group["lr"] = lr
                else:
                    log.error(f"Optimizer index {i} out of range, skipping setting LR.")

        trainer.strategy.barrier()

        if trainer.is_global_zero:
            log.info(f"LR set to {lr} on all processes, removing {self.lr_file}")
            os.remove(self.lr_file)

    def state_dict(self) -> dict[str, Any]:
        return {"lr_file": str(self.lr_file)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "lr_file" in state_dict:
            self.lr_file = pathlib.Path(state_dict["lr_file"])

        state_dict.pop("lr_file", None)
