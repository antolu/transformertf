from __future__ import annotations

import logging
import typing

import lightning as L
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import BaseTransform, EncoderDecoderDataset, TimeSeriesDataset
from ..data._covariates import TIME_PREFIX as TIME
from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class PlotHysteresisCallback(L.pytorch.callbacks.callback.Callback):
    """
    Callback to plot the hysteresis of the model on the validation set.
    """

    def __init__(self, plot_every: int = 1):
        super().__init__()
        self.plot_every = plot_every

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningModuleBase
    ) -> None:
        self.plot_and_log(trainer, pl_module)

    def plot_and_log(self, trainer: L.Trainer, pl_module: LightningModuleBase) -> None:
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

        val_dataset = typing.cast(EncoderDecoderDataset, val_dataloader.dataset)
        validation_outputs = pl_module.validation_outputs

        predictions = torch.cat([
            o["point_prediction"].squeeze() for o in validation_outputs[0]
        ])

        if len(predictions) > val_dataset.num_points:
            predictions = predictions[
                : val_dataset.num_points - val_dataset.ctxt_seq_len
            ]

        if isinstance(val_dataset, TimeSeriesDataset):
            indices = slice(0, len(predictions))
            sample_len = val_dataset.seq_len
        elif isinstance(val_dataset, EncoderDecoderDataset):
            indices = slice(
                val_dataset.ctxt_seq_len,
                val_dataset.ctxt_seq_len + len(predictions)
                if len(predictions) < val_dataset.num_points
                else val_dataset.num_points,
            )
            sample_len = val_dataset.tgt_seq_len
        else:
            msg = "Only TimeSeriesDataset and EncoderDecoderDataset are supported."
            raise ValueError(msg)  # noqa: TRY004

        try:
            time = val_dataset._sample_gen[0]._input_data[TIME].to_numpy()  # noqa: SLF001
            time = time[indices]
            time = time - time[0]
        except KeyError:
            time = np.arange(len(predictions))

        targets = val_dataset._sample_gen[0]._label_data.to_numpy().flatten()  # noqa: SLF001
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
        depends_on = val_dataset._sample_gen[0]._input_data[depends_on_key].to_numpy()  # noqa: SLF001
        depends_on = depends_on[indices]
        depends_on_key = depends_on_key.split("__")[-1]
        depends_on = (
            val_dataset.transforms[depends_on_key].inverse_transform(depends_on).numpy()
        )

        target_transform = transforms[target_key]
        if target_transform.transform_type == BaseTransform.TransformType.XY:
            predictions = (
                target_transform.inverse_transform(depends_on, predictions)
                .cpu()
                .numpy()
            )
            targets = (
                target_transform.inverse_transform(depends_on, targets).cpu().numpy()
            )
        else:
            predictions = target_transform.inverse_transform(predictions).cpu().numpy()
            targets = target_transform.inverse_transform(targets).cpu().numpy()

        prediction_horizons = np.arange(sample_len, len(predictions), sample_len)

        fig = plot_hysteresis_phase_space(depends_on, predictions, targets)
        trainer.logger.experiment.add_figure(
            "hysteresis/validation", fig, global_step=trainer.global_step
        )

        fig = plot_field_curve(time, predictions, targets, prediction_horizons)
        trainer.logger.experiment.add_figure(
            "field_curve/validation", fig, global_step=trainer.global_step
        )

        plt.close("all")


def plot_hysteresis_phase_space(
    current: np.ndarray, field_pred: np.ndarray, field_gt: np.ndarray
) -> matplotlib.figure.Figure:
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
    """Plot the true and predicted field over time."""
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
