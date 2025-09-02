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


class PlotBdotHysteresisCallback(L.pytorch.callbacks.callback.Callback):
    """
    Lightning callback for plotting bdot (magnetic field derivative) hysteresis curves.

    This callback generates and logs visualizations of magnetic field derivative
    behavior during validation. It creates plots for both current vs bdot and
    current derivative vs bdot relationships, which are crucial for understanding
    dynamic magnetic field behavior in accelerator magnets.

    The callback creates three types of plots:
    1. Phase space plot showing bdot prediction errors vs. current (i vs bdot)
    2. Phase space plot showing bdot prediction errors vs. current derivative (idot vs bdot)
    3. Time series plot comparing predicted and ground truth bdot values

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
        Main plotting logic that generates and logs bdot hysteresis visualizations.

    Notes
    -----
    - Requires single validation dataloader (multiple dataloaders not supported)
    - Works with TimeSeriesDataset and EncoderDecoderDataset types
    - Assumes current (i) is the first covariate after time (if applicable)
    - Assumes current derivative (idot) is the second covariate after time
    - Automatically applies inverse transforms to show plots in original units
    - Supports TensorBoard and Neptune loggers only
    - Plots are logged with global step for proper timeline tracking
    - All matplotlib figures are properly closed to prevent memory leaks

    Examples
    --------
    Plot bdot hysteresis every epoch during training:

    >>> callback = PlotBdotHysteresisCallback(plot_every=1)
    >>> trainer = L.Trainer(callbacks=[callback])

    Reduce plotting frequency for long training runs:

    >>> callback = PlotBdotHysteresisCallback(plot_every=10)
    >>> trainer = L.Trainer(callbacks=[callback])

    Integration with other callbacks:

    >>> callbacks = [
    ...     PlotBdotHysteresisCallback(plot_every=5),
    ...     L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss"),
    ... ]
    >>> trainer = L.Trainer(
    ...     callbacks=callbacks,
    ...     logger=L.pytorch.loggers.TensorBoardLogger("logs/")
    ... )

    See Also
    --------
    PlotHysteresisCallback : For standard B-field hysteresis plotting
    transformertf.data.TimeSeriesDataset : Compatible dataset type
    transformertf.data.EncoderDecoderDataset : Compatible dataset type
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

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningModuleBase
    ) -> None:
        """
        Generate and log bdot hysteresis plots at the end of validation epoch.

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
        Generate bdot hysteresis plots and log them to the experiment tracker.

        Creates three visualization types:
        1. Phase space plot showing bdot prediction errors vs. current (i)
        2. Phase space plot showing bdot prediction errors vs. current derivative (idot)
        3. Time series plot comparing predicted and ground truth bdot values

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
                "The PlotBdotHysteresisCallback only supports a single "
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

        # Get target key (bdot)
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

        # Get input columns (excluding time columns)
        input_columns = [
            col
            for col in val_dataset._sample_gen[0]._input_data.columns  # noqa: SLF001
            if not col.startswith("__time__")
        ]

        # Assume first column after time is current (i), second is current derivative (idot)
        if len(input_columns) < 2:
            msg = (
                f"Expected at least 2 input columns (i, idot), got {len(input_columns)}"
            )
            raise ValueError(msg)

        current_key = input_columns[0]  # First non-time column (i)
        current_dot_key = input_columns[1]  # Second non-time column (idot)

        # Get column indices
        current_idx = val_dataset._sample_gen[0]._input_data.columns.get_loc(  # noqa: SLF001
            current_key
        )
        current_dot_idx = val_dataset._sample_gen[0]._input_data.columns.get_loc(  # noqa: SLF001
            current_dot_key
        )

        # Extract current and current derivative data
        if isinstance(val_dataset, TimeSeriesDataset):
            current = torch.cat([
                o["input"][..., current_idx].squeeze() for o in validation_outputs[0]
            ])
            current_dot = torch.cat([
                o["input"][..., current_dot_idx].squeeze()
                for o in validation_outputs[0]
            ])
        elif isinstance(val_dataset, EncoderDecoderDataset):
            current = torch.cat([
                o["decoder_input"][..., current_idx].squeeze()
                for o in validation_outputs[0]
            ])
            current_dot = torch.cat([
                o["decoder_input"][..., current_dot_idx].squeeze()
                for o in validation_outputs[0]
            ])

        current = current[indices]
        current_dot = current_dot[indices]

        # Apply inverse transforms
        current_key_clean = current_key.split("__")[-1]
        current_dot_key_clean = current_dot_key.split("__")[-1]

        current = (
            val_dataset.transforms[current_key_clean].inverse_transform(current).numpy()
        )
        current_dot = (
            val_dataset.transforms[current_dot_key_clean]
            .inverse_transform(current_dot)
            .numpy()
        )

        target_transform = transforms[target_key]

        # Apply inverse transform to targets and predictions
        if self._has_delta_transform(target_transform):
            # Handle delta transform case
            if target_transform.transform_type == BaseTransform.TransformType.XY:
                predictions_transformed = target_transform.inverse_transform(
                    current, predictions
                )
                targets_transformed = target_transform.inverse_transform(
                    current, targets
                )
            else:
                predictions_transformed = target_transform.inverse_transform(
                    predictions
                )
                targets_transformed = target_transform.inverse_transform(targets)

            predictions = predictions_transformed.cpu().numpy()
            targets = targets_transformed.cpu().numpy()
        else:
            # Standard inverse transform
            if target_transform.transform_type == BaseTransform.TransformType.XY:
                predictions = (
                    target_transform.inverse_transform(current, predictions)
                    .cpu()
                    .numpy()
                )
                targets = (
                    target_transform.inverse_transform(current, targets).cpu().numpy()
                )
            else:
                predictions = (
                    target_transform.inverse_transform(predictions).cpu().numpy()
                )
                targets = target_transform.inverse_transform(targets).cpu().numpy()

        time = np.arange(len(predictions))
        prediction_horizons = np.arange(sample_len, len(predictions), sample_len)

        # Create plots
        fig_i_bdot = plot_bdot_phase_space(
            current, predictions, targets, "Current [A]", "i vs bdot"
        )
        fig_idot_bdot = plot_bdot_phase_space(
            current_dot,
            predictions,
            targets,
            "Current derivative [A/s]",
            "idot vs bdot",
        )
        fig_bdot_time = plot_bdot_time_series(
            time, predictions, targets, prediction_horizons
        )

        # Log plots
        if isinstance(trainer.logger, L.pytorch.loggers.TensorBoardLogger):
            trainer.logger.experiment.add_figure(
                "bdot_time_series/validation",
                fig_bdot_time,
                global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_figure(
                "bdot_hysteresis_i/validation",
                fig_i_bdot,
                global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_figure(
                "bdot_hysteresis_idot/validation",
                fig_idot_bdot,
                global_step=trainer.global_step,
            )
        elif isinstance(trainer.logger, L.pytorch.loggers.neptune.NeptuneLogger):
            trainer.logger.experiment["validation/bdot_time_series"].append(
                fig_bdot_time
            )
            trainer.logger.experiment["validation/bdot_hysteresis_i"].append(fig_i_bdot)
            trainer.logger.experiment["validation/bdot_hysteresis_idot"].append(
                fig_idot_bdot
            )
            trainer.logger.experiment.sync()
        else:
            msg = "The PlotBdotHysteresisCallback only supports TensorBoard and Neptune loggers."
            raise ValueError(msg)  # noqa: TRY004

        plt.close("all")


def plot_bdot_phase_space(
    x_values: np.ndarray,
    bdot_pred: np.ndarray,
    bdot_gt: np.ndarray,
    x_label: str,
    title_suffix: str,
) -> matplotlib.figure.Figure:
    """
    Create bdot phase space plot showing prediction errors vs. x-axis variable.

    Generates a two-panel figure showing:
    1. Top panel: Difference between ground truth and predicted bdot
       plotted against x-axis variable (current or current derivative)
    2. Bottom panel: Histogram of absolute error distribution weighted by x-values

    Parameters
    ----------
    x_values : np.ndarray
        X-axis values (current or current derivative).
    bdot_pred : np.ndarray
        Predicted magnetic field derivative values in T/s.
    bdot_gt : np.ndarray
        Ground truth magnetic field derivative values in T/s.
    x_label : str
        Label for x-axis (e.g., "Current [A]" or "Current derivative [A/s]").
    title_suffix : str
        Suffix for plot title (e.g., "i vs bdot" or "idot vs bdot").

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the bdot phase space visualization.

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
        x_values,
        (bdot_gt - bdot_pred) * 1e4,
        linewidth=0.15,
        c="firebrick",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Bdot difference [$10^{-4}$ T/s]")
    ax1.set_title(
        f"Difference between ground truth and predicted bdot - {title_suffix}"
    )

    ax2.hist(
        x_values,
        bins=50,
        weights=np.abs(bdot_gt - bdot_pred),
        density=True,
        log=True,
        color="firebrick",
    )
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Density")
    ax2.set_title("Absolute error distribution")

    fig.tight_layout()

    return fig


def plot_bdot_time_series(
    time: np.ndarray,
    bdot_pred: np.ndarray,
    bdot_gt: np.ndarray,
    prediction_horizons: np.ndarray | None = None,
) -> matplotlib.figure.Figure:
    """
    Create time series plot comparing predicted and ground truth bdot values.

    Generates a two-panel figure showing:
    1. Top panel: Difference between ground truth and predicted bdot over time
    2. Bottom panel: Overlay of ground truth and predicted bdot time series

    Parameters
    ----------
    time : np.ndarray
        Time values in seconds for the x-axis.
    bdot_pred : np.ndarray
        Predicted magnetic field derivative values in T/s.
    bdot_gt : np.ndarray
        Ground truth magnetic field derivative values in T/s.
    prediction_horizons : np.ndarray, optional
        Time indices marking prediction horizons. If provided, these points
        are highlighted with scatter markers on the bottom panel.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the bdot time series visualization.

    Notes
    -----
    - Error differences are scaled by 1e4 for better visualization
    - Top panel includes ±1e-4 T/s reference lines for error assessment
    - Y-axis range for error plot is limited to ±3e-4 T/s
    - Legend distinguishes between ground truth (Bdot) and prediction (zdot)
    - Figure layout is optimized with tight_layout()
    """
    s = slice(None, None)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(12, 6))

    ax2.plot(
        time[s],
        bdot_gt[s],
        label="$\\dot{B}$ (ground truth)",
        c="firebrick",
    )
    ax2.plot(
        time[s],
        bdot_pred[s],
        label="$\\dot{z}$ (prediction)",
        linestyle="-",
        c="midnightblue",
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Field derivative [T/s]")
    ax2.set_title("Ground truth bdot vs. predicted bdot")
    if prediction_horizons is not None:
        ax2.scatter(
            time[prediction_horizons],
            bdot_gt[prediction_horizons],
            label="Prediction horizon",
            c="k",
            marker=".",
        )
    ax2.legend()

    ax1.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
    ax1.plot(
        time[s],
        (bdot_gt[s] - bdot_pred[s]) * 1e4,
        c="orangered",
        label=r"$\hat{\dot{B}} - \dot{B}$",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Field derivative [$10^{-4}$ T/s]")
    ax1.set_title("Difference between ground truth and predicted bdot")
    ax1.legend()

    ax1.axhline(y=1, color="k", linestyle="dotted", linewidth=0.8)
    ax1.axhline(y=-1, color="k", linestyle="dotted", linewidth=0.8)
    ax1.set_ylim(-3, 3)

    fig.tight_layout()

    return fig
