from __future__ import annotations

import logging
import typing

import lightning as L
import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import BaseTransform, EncoderDecoderDataset, TimeSeriesDataset
from ..data.transform import DeltaTransform
from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class PlotHysteresisCallback(L.pytorch.callbacks.callback.Callback):
    """
    Lightning callback for plotting hysteresis curves and field predictions.

    This callback generates and logs visualizations of magnetic field hysteresis
    behavior during validation. It creates two types of plots:
    1. Hysteresis phase space plot showing prediction errors vs. current
    2. Time series plot comparing predicted and ground truth magnetic fields

    The callback is designed for magnetic field modeling tasks where understanding
    hysteresis behavior is crucial for model evaluation. It supports both
    TensorBoard and Neptune loggers for experiment tracking.

    Parameters
    ----------
    plot_every : int, default=1
        Frequency of plot generation in epochs. For example:
        - 1: Plot every epoch
        - 5: Plot every 5 epochs
        - 10: Plot every 10 epochs
        Use higher values to reduce computational overhead during training.

    Attributes
    ----------
    plot_every : int
        Frequency of plot generation in epochs.

    Methods
    -------
    on_validation_epoch_end(trainer, pl_module)
        Called at the end of each validation epoch to generate plots.
    plot_and_log(trainer, pl_module)
        Main plotting logic that generates and logs hysteresis visualizations.

    Notes
    -----
    - Requires single validation dataloader (multiple dataloaders not supported)
    - Works with TimeSeriesDataset and EncoderDecoderDataset types
    - Automatically applies inverse transforms to show plots in original units
    - Supports TensorBoard and Neptune loggers only
    - Plots are logged with global step for proper timeline tracking
    - All matplotlib figures are properly closed to prevent memory leaks

    Examples
    --------
    Plot every epoch during training:

    >>> callback = PlotHysteresisCallback(plot_every=1)
    >>> trainer = L.Trainer(callbacks=[callback])

    Reduce plotting frequency for long training runs:

    >>> callback = PlotHysteresisCallback(plot_every=10)
    >>> trainer = L.Trainer(callbacks=[callback])

    Integration with logging and model saving:

    >>> callbacks = [
    ...     PlotHysteresisCallback(plot_every=5),
    ...     L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss"),
    ...     LogHparamsCallback(monitor="val_loss", mode="min")
    ... ]
    >>> trainer = L.Trainer(
    ...     callbacks=callbacks,
    ...     logger=L.pytorch.loggers.TensorBoardLogger("logs/")
    ... )

    See Also
    --------
    transformertf.data.TimeSeriesDataset : Compatible dataset type
    transformertf.data.EncoderDecoderDataset : Compatible dataset type
    lightning.pytorch.loggers.TensorBoardLogger : Supported logger
    lightning.pytorch.loggers.NeptuneLogger : Supported logger
    """

    def __init__(self, plot_every: int = 1):
        super().__init__()
        self.plot_every = plot_every

    def _has_delta_transform(self, transform_collection):
        """Check if transform collection contains DeltaTransform."""
        for transform in transform_collection.transforms:
            if isinstance(transform, DeltaTransform):
                return True
        return False

    def _get_time_step_size(self, val_dataset):
        """Extract actual time step size (dt) from dataset time information."""
        try:
            # Access raw time data from sample generator
            input_data = val_dataset._sample_gen[0]._input_data  # noqa: SLF001

            # Find time column (prefixed with TIME_PREFIX = "__time__")
            time_columns = [
                col for col in input_data.columns if col.startswith("__time__")
            ]

            if time_columns:
                time_col = time_columns[0]  # Use first time column
                time_values = input_data[time_col].values

                # Calculate dt as median difference (robust to outliers)
                dt = np.median(np.diff(time_values))
                return float(dt)
            # No time column found, assume unit time steps
            return 1.0  # noqa: TRY300

        except Exception:
            # Fallback to unit time steps if extraction fails
            return 1.0

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningModuleBase
    ) -> None:
        """
        Generate and log hysteresis plots at the end of validation epoch.

        Called automatically by Lightning after each validation epoch.
        Delegates to plot_and_log method for the actual plotting logic.

        Parameters
        ----------
        trainer : L.Trainer
            Lightning trainer instance containing validation data and logger.
        pl_module : LightningModuleBase
            Lightning module with validation outputs for plotting.
        """
        self.plot_and_log(trainer, pl_module)

    def plot_and_log(self, trainer: L.Trainer, pl_module: LightningModuleBase) -> None:
        """
        Generate hysteresis plots and log them to the experiment tracker.

        Creates two visualization types:
        1. Hysteresis phase space plot showing prediction errors vs. input current
        2. Time series plot comparing predicted and ground truth magnetic fields

        The method handles data extraction from validation outputs, applies inverse
        transforms to convert back to original units, and logs the resulting plots
        to the configured logger (TensorBoard or Neptune).

        Parameters
        ----------
        trainer : L.Trainer
            Lightning trainer containing validation dataloader and logger.
        pl_module : LightningModuleBase
            Lightning module containing validation outputs and predictions.

        Raises
        ------
        ValueError
            If multiple validation dataloaders are provided (not supported).
        ValueError
            If dataset type is not TimeSeriesDataset or EncoderDecoderDataset.
        ValueError
            If logger type is not TensorBoard or Neptune.

        Notes
        -----
        - Only executes on global rank 0 process in distributed training
        - Respects plot_every frequency setting to control computational overhead
        - Automatically extracts target and input data from validation outputs
        - Applies dataset-specific inverse transforms for proper visualization
        - Logs plots with current global step for timeline tracking
        - Closes all matplotlib figures to prevent memory leaks
        """
        if not trainer.is_global_zero:
            return

        if trainer.current_epoch % self.plot_every != 0:
            return

        val_dataloader = trainer.val_dataloaders

        if isinstance(val_dataloader, list):
            msg = (
                "The PlotHysteresisCallback only supports a single "
                "validation dataloader."
            )
            raise ValueError(msg)  # noqa: TRY004

        val_dataset = typing.cast(
            EncoderDecoderDataset | TimeSeriesDataset, val_dataloader.dataset
        )
        validation_outputs = pl_module.validation_outputs

        predictions = torch.cat([
            o["point_prediction"].squeeze() for o in validation_outputs[0]
        ])

        if isinstance(val_dataset, TimeSeriesDataset):
            indices = slice(0, len(predictions))
            sample_len = val_dataset.seq_len
        elif isinstance(val_dataset, EncoderDecoderDataset):
            indices = slice(
                0,
                len(predictions)
                if len(predictions) < val_dataset.num_points
                else val_dataset.num_points,
            )
            sample_len = val_dataset.tgt_seq_len
        else:
            msg = "Only TimeSeriesDataset and EncoderDecoderDataset are supported."
            raise ValueError(msg)  # noqa: TRY004

        assert val_dataset._sample_gen[0]._label_data is not None  # noqa: SLF001
        targets = torch.cat([o["target"].squeeze() for o in validation_outputs[0]])
        targets = targets[indices]  # type: ignore[assignment]

        transforms = val_dataset.transforms
        try:
            target_key = next(
                filter(
                    lambda x: x.startswith("__target__"),
                    val_dataset._sample_gen[0]._label_data.columns,  # noqa: SLF001
                )
            )
            target_key = target_key.split("__")[-1]
        except (StopIteration, KeyError) as e:
            msg = f"Target key not found in transforms with keys {transforms.keys()}"
            raise ValueError(msg) from e

        depends_on_key = next(
            filter(
                lambda x: not x.startswith("__time__") and not x.endswith("_dot"),
                val_dataset._sample_gen[0]._input_data.columns,  # noqa: SLF001
            )
        )
        # find which df index the depends_on_key is in
        depends_on_idx = val_dataset._sample_gen[0]._input_data.columns.get_loc(  # noqa: SLF001
            depends_on_key
        )

        if isinstance(val_dataset, TimeSeriesDataset):
            depends_on = torch.cat([
                o["input"][..., depends_on_idx].squeeze() for o in validation_outputs[0]
            ])
        elif isinstance(val_dataset, EncoderDecoderDataset):
            depends_on = torch.cat([
                o["decoder_input"][..., depends_on_idx].squeeze()
                for o in validation_outputs[0]
            ])
        else:
            msg = "Only TimeSeriesDataset and EncoderDecoderDataset are supported."
            raise RuntimeError(msg)  # noqa: TRY004

        depends_on = depends_on[indices]

        depends_on_key = depends_on_key.split("__")[-1]
        depends_on = (
            val_dataset.transforms[depends_on_key].inverse_transform(depends_on).numpy()
        )

        target_transform = transforms[target_key]

        if self._has_delta_transform(target_transform):
            # Extract time step size for proper scaling
            # Apply inverse transform (includes cumsum from DeltaTransform)
            if target_transform.transform_type == BaseTransform.TransformType.XY:
                predictions_cumsum = target_transform.inverse_transform(
                    depends_on, predictions
                )
                targets_cumsum = target_transform.inverse_transform(depends_on, targets)
            else:
                predictions_cumsum = target_transform.inverse_transform(predictions)
                targets_cumsum = target_transform.inverse_transform(targets)

            # Use first target value as integration constant
            first_target = targets_cumsum[0].item()

            # Proper reconstruction: y[t] = y[0] + cumsum(delta * dt)
            # Since inverse_transform already did cumsum, we need to:
            # 1. Scale by dt (cumsum result needs dt scaling)
            # 2. Adjust starting point to first_target

            predictions_scaled = predictions_cumsum
            targets_scaled = targets_cumsum

            # Adjust to start from correct integration constant
            predictions = (
                (predictions_scaled - predictions_scaled[0] + first_target)
                .cpu()
                .numpy()
            )
            targets = targets_scaled.cpu().numpy()

            # Update time array to reflect actual time values
            time = np.arange(len(predictions))

        else:
            # Standard inverse transform (existing logic)
            if target_transform.transform_type == BaseTransform.TransformType.XY:
                predictions = (
                    target_transform.inverse_transform(depends_on, predictions)
                    .cpu()
                    .numpy()
                )
                targets = (
                    target_transform.inverse_transform(depends_on, targets)
                    .cpu()
                    .numpy()
                )
            else:
                predictions = (
                    target_transform.inverse_transform(predictions).cpu().numpy()
                )
                targets = target_transform.inverse_transform(targets).cpu().numpy()

            # Keep existing time array
            time = np.arange(len(predictions))

        prediction_horizons = np.arange(sample_len, len(predictions), sample_len)

        fig_hysteresis = plot_hysteresis_phase_space(depends_on, predictions, targets)

        fig_field = plot_field_curve(time, predictions, targets, prediction_horizons)
        if isinstance(trainer.logger, L.pytorch.loggers.TensorBoardLogger):
            trainer.logger.experiment.add_figure(
                "field_curve/validation", fig_field, global_step=trainer.global_step
            )
            trainer.logger.experiment.add_figure(
                "hysteresis/validation", fig_hysteresis, global_step=trainer.global_step
            )
        elif isinstance(trainer.logger, L.pytorch.loggers.neptune.NeptuneLogger):
            trainer.logger.experiment["validation/field_curve"].append(fig_field)
            trainer.logger.experiment["validation/hysteresis"].append(fig_hysteresis)
            trainer.logger.experiment.sync()
        else:
            msg = "The PlotHysteresisCallback only supports TensorBoard and Neptune loggers."
            raise ValueError(msg)  # noqa: TRY004

        plt.close("all")


def plot_hysteresis_phase_space(
    current: np.ndarray, field_pred: np.ndarray, field_gt: np.ndarray
) -> matplotlib.figure.Figure:
    """
    Create hysteresis phase space plot showing prediction errors vs. current.

    Generates a two-panel figure showing:
    1. Top panel: Difference between ground truth and predicted magnetic field
       plotted against input current, revealing hysteresis loop structure
    2. Bottom panel: Histogram of absolute error distribution weighted by current

    Parameters
    ----------
    current : np.ndarray
        Input current values in Amperes.
    field_pred : np.ndarray
        Predicted magnetic field values in Tesla.
    field_gt : np.ndarray
        Ground truth magnetic field values in Tesla.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the hysteresis phase space visualization.

    Notes
    -----
    - Error differences are scaled by 1e4 for better visualization
    - Histogram uses logarithmic scale for error distribution
    - Figure layout is optimized with tight_layout()
    """
    ax1: plt.Axes
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex="all", figsize=(8, 6), height_ratios=[4, 1]
    )
    ax1.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
    ax1.axvline(x=0, color="k", linestyle="--", linewidth=0.8)
    ax1.plot(
        current,
        (field_gt - field_pred) * 1e4,
        linewidth=0.15,
        c="firebrick",
    )
    ax1.set_xlabel("Current [A]")
    ax1.set_ylabel("Transfer function difference [$10^{-4}$ T]")
    ax1.set_title("Difference between ground truth and predicted B-field")

    ax2.hist(
        current,
        bins=50,
        weights=np.abs(field_gt - field_pred),
        density=True,
        log=True,
        color="firebrick",
    )
    ax2.set_xlabel("Current [A]")
    ax2.set_ylabel("Density")
    ax2.set_title("Absolute error distribution")

    fig.tight_layout()

    return fig


def plot_field_curve(
    time: np.ndarray,
    field_pred: np.ndarray,
    field_gt: np.ndarray,
    prediction_horizons: np.ndarray | None = None,
) -> matplotlib.figure.Figure:
    """
    Create time series plot comparing predicted and ground truth magnetic fields.

    Generates a two-panel figure showing:
    1. Top panel: Difference between ground truth and predicted field over time
    2. Bottom panel: Overlay of ground truth and predicted field time series

    Parameters
    ----------
    time : np.ndarray
        Time values in seconds for the x-axis.
    field_pred : np.ndarray
        Predicted magnetic field values in Tesla.
    field_gt : np.ndarray
        Ground truth magnetic field values in Tesla.
    prediction_horizons : np.ndarray, optional
        Time indices marking prediction horizons. If provided, these points
        are highlighted with scatter markers on the bottom panel.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the field curve time series visualization.

    Notes
    -----
    - Error differences are scaled by 1e4 for better visualization
    - Top panel includes ±1e-4 Tesla reference lines for error assessment
    - Y-axis range for error plot is limited to ±3e-4 Tesla
    - Legend distinguishes between ground truth (B) and prediction (z)
    - Figure layout is optimized with tight_layout()
    """
    s = slice(None, None)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(12, 6))

    ax2.plot(
        time[s],
        field_gt[s],
        label="$B$ (ground truth)",
        c="firebrick",
    )
    ax2.plot(
        time[s],
        field_pred[s],
        label="$z$ (prediction)",
        linestyle="-",
        c="midnightblue",
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Field [T]")
    ax2.set_title("Ground truth B-field vs. predicted B-field")
    if prediction_horizons is not None:
        ax2.scatter(
            time[prediction_horizons],
            field_gt[prediction_horizons],
            label="Prediction horizon",
            c="k",
            marker=".",
        )
    # ax2.set_xlim(data.time[0], data.time[-1])
    ax2.legend()

    ax1.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
    ax1.plot(
        time[s],
        (field_gt[s] - field_pred[s]) * 1e4,
        c="orangered",
        label=r"$\hat{B} - B$",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Field [$10^{-4}$ T]")
    ax1.set_title("Difference between ground truth and predicted B-field")
    ax1.legend()

    ax1.axhline(y=1, color="k", linestyle="dotted", linewidth=0.8)
    ax1.axhline(y=-1, color="k", linestyle="dotted", linewidth=0.8)
    ax1.set_ylim(-3, 3)

    fig.tight_layout()

    return fig
