from __future__ import annotations

import logging

import lightning as L

from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class SetOptimizerParamsCallback(L.pytorch.callbacks.Callback):
    """
    Callback to set optimizer parameters on train start. This is useful for
    the population based training algorithm when resuming from a checkpoint, as
    the optimizer parameters are not loaded into the lightning module or data module,
    but only by the Trainer when a checkpoint is loaded.
    """

    def __init__(
        self,
        lr: float | None = None,
        momentum: float | None = None,
        weight_decay: float | None = None,
        **kwargs: float,
    ):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.params = {
            key: value for key, value in kwargs.items() if isinstance(value, float)
        }

        remaining_keys = set(kwargs.keys()) - set(self.params.keys())
        if remaining_keys:
            msg = f"Unsupported optimizer parameters: {remaining_keys}"
            raise ValueError(msg)

    def on_train_start(
        self, trainer: L.Trainer, pl_module: LightningModuleBase
    ) -> None:
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_scheduler_configs

        if len(optimizers) > 1:
            msg = f"The {self.__class__.__name__} callback only supports a single optimizer."
            raise ValueError(msg)

        if len(lr_schedulers) > 1:
            msg = f"The {self.__class__.__name__} callback only supports a single LR scheduler."
            raise ValueError(msg)

        optimizer = optimizers[0]
        lr_scheduler = lr_schedulers[0].scheduler

        for param_group in optimizer.param_groups:
            if self.lr is not None and "lr" in param_group:
                log.debug(f"Setting LR to {self.lr}.")
                param_group["lr"] = self.lr
            if self.momentum is not None and "momentum" in param_group:
                log.debug(f"Setting momentum to {self.momentum}.")
                param_group["momentum"] = self.momentum
            if self.weight_decay is not None and "weight_decay" in param_group:
                log.debug(f"Setting weight decay to {self.weight_decay}.")
                param_group["weight_decay"] = self.weight_decay
            for key, value in self.params.items():
                param_group[key] = value

            for param_name, param_value in self.params.items():
                if param_name not in param_group:
                    continue

                log.debug(f"Setting optimizer parameter {param_name} to {param_value}.")
                param_group[param_name] = param_value

        for param_name, param_value in self.params.items():
            if not hasattr(lr_scheduler, param_name):
                continue

            log.debug(f"Setting LR scheduler parameter {param_name} to {param_value}.")
            setattr(lr_scheduler, param_name, param_value)
