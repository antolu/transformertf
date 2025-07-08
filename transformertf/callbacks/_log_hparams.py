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
    """
    Lightning callback to log hyperparameters with metrics when validation improves.

    This callback addresses a limitation in TensorBoard logging where hyperparameters
    are typically logged at training start before metrics are available. Instead,
    this callback logs hyperparameters alongside validation metrics whenever the
    monitored metric improves, creating meaningful hyperparameter-metric associations
    for experiment tracking and hyperparameter optimization.

    The callback is specifically designed for TensorBoard logger compatibility and
    will be skipped for other logger types that handle hyperparameter logging
    more intelligently.

    Parameters
    ----------
    monitor : str
        Name of the metric to monitor for improvement. Should match a metric
        name available in `trainer.callback_metrics` (e.g., 'val_loss',
        'val_accuracy'). The metric determines when hyperparameters are logged.
    mode : {"min", "max"}, default="min"
        Direction of metric improvement:
        - "min": Log when metric decreases (e.g., for loss)
        - "max": Log when metric increases (e.g., for accuracy)

    Attributes
    ----------
    monitor : str
        The metric being monitored for improvement.
    mode : str
        The direction of improvement ("min" or "max").
    last_metric_value : float or None
        The last recorded value of the monitored metric. Used to determine
        if the current metric represents an improvement.

    Methods
    -------
    on_validation_epoch_end(trainer, module)
        Called at the end of each validation epoch to potentially log hyperparameters.

    Notes
    -----
    - Only works with TensorBoardLogger; other loggers are automatically skipped
    - Hyperparameters include all model hyperparameters plus model parameter count
    - Metrics are filtered to validation metrics only (those ending with "validation")
    - The callback maintains state to track metric improvements across epochs

    Examples
    --------
    Basic usage for monitoring validation loss:

    >>> callback = LogHparamsCallback(monitor="val_loss", mode="min")
    >>> trainer = L.Trainer(callbacks=[callback])

    Monitoring validation accuracy:

    >>> callback = LogHparamsCallback(monitor="val_accuracy", mode="max")
    >>> trainer = L.Trainer(callbacks=[callback])

    Integration with other callbacks:

    >>> callbacks = [
    ...     LogHparamsCallback(monitor="val_loss", mode="min"),
    ...     L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
    ...     L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    ... ]
    >>> trainer = L.Trainer(callbacks=callbacks)

    See Also
    --------
    lightning.pytorch.callbacks.ModelCheckpoint : For saving best models
    lightning.pytorch.callbacks.EarlyStopping : For stopping on metric plateau
    lightning.pytorch.loggers.TensorBoardLogger : Required logger type
    """

    def __init__(self, monitor: str, mode: typing.Literal["min", "max"] = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

        self.last_metric_value: float | None = None

    def _should_log_hparams(self, trainer: L.Trainer) -> bool:
        """
        Determine if hyperparameters should be logged based on metric improvement.

        Parameters
        ----------
        trainer : L.Trainer
            Lightning trainer containing callback metrics.

        Returns
        -------
        bool
            True if the monitored metric has improved and hyperparameters
            should be logged, False otherwise.
        """
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
        """
        Log hyperparameters with metrics if validation metric improved.

        Called automatically by Lightning at the end of each validation epoch.
        Checks if the monitored metric has improved and logs hyperparameters
        with current validation metrics to TensorBoard if so.

        Parameters
        ----------
        trainer : L.Trainer
            Lightning trainer instance containing logger and metrics.
        module : LightningModuleBase
            Lightning module containing hyperparameters to log.

        Notes
        -----
        - Only logs with TensorBoardLogger; other loggers are skipped
        - Updates internal state to track the best metric value
        - Includes model parameter count in logged hyperparameters
        - Filters metrics to validation-only (names ending with "validation")
        """
        if trainer.logger is None:
            return

        # we only need this logger with TensorBoardLogger; other loggers are more intelligent...
        if not isinstance(trainer.logger, L.pytorch.loggers.TensorBoardLogger):
            return

        if not self._should_log_hparams(trainer):
            return

        metrics = {
            k.split("/")[0]: v  # remove the "validation" prefix
            for k, v in trainer.callback_metrics.items()
            if k.endswith("validation")
        }
        hparams = dict(module.hparams.items())
        hparams["num_params"] = sum(p.numel() for p in module.parameters())

        trainer.logger.log_hyperparams(hparams, metrics=metrics)
