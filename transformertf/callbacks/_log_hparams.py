"""
Implementation of Lightning callback to log hparams with metrics when the
metric improves on validation end. This is because the tensorboard logger
allows logging metrics with the hyperparameters, but this is normally done
at the start of training where metrics are not available yet. This callback
logs the hparams with the metrics when the metric improves on validation end.
The hparams are logged with the metric value and the epoch where the metric
improved.
"""

from __future__ import annotations

import logging
import typing

import lightning as L

from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class LogHparamsCallback(L.pytorch.callbacks.callback.Callback):
    def __init__(self, monitor: str, mode: typing.Literal["min", "max"] = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

        self.last_metric_value: float | None = None

    def _should_log_hparams(self, trainer: L.Trainer) -> bool:
        if self.monitor not in trainer.callback_metrics:
            return False

        metric = trainer.callback_metrics[self.monitor]

        if self.last_metric_value is None:
            self.last_metric_value = metric
            return True

        if self.mode == "min":
            return metric < self.last_metric_value

        return metric > self.last_metric_value

    def on_validation_epoch_end(
        self, trainer: L.Trainer, module: LightningModuleBase
    ) -> None:
        if trainer.logger is None:
            return

        if not self._should_log_hparams(trainer):
            return

        metrics = {
            k.split("/")[0]: v  # remove the "validation" prefix
            for k, v in trainer.callback_metrics.items()
            if k.endswith("validation")
        }

        trainer.logger.log_hyperparams(module.hparams, metrics=metrics)
