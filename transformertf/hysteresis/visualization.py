from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import torch

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .base import BaseHysteresis


def plot_hysterion_density(
    H: BaseHysteresis,
    density: torch.Tensor | None = None,
    ax: Axes | None = None,
) -> tuple[plt.Figure, Axes, Axes]:
    new_ax = ax is None
    if new_ax:
        fig, ax = plt.subplots()
    else:
        fig = None

    x = H.mesh_points[:, 0]
    y = H.mesh_points[:, 1]

    if density is None:
        density = H.hysterion_density.detach()

    assert ax is not None
    den = density  # * H.get_mesh_size(x, y)
    c = ax.tripcolor(x, y, den)
    if new_ax:
        assert fig is not None
        fig.colorbar(c)
        return fig, ax, c
    return fig, ax, c


def plot_bayes_predicition(
    summary: dict[str, dict[str, torch.Tensor]],
    m: torch.Tensor,
    baseline: bool = False,
) -> tuple[plt.Figure, Axes]:
    y = summary["obs"]
    ax: Axes
    fig, ax = plt.subplots()
    ax.plot(m.detach(), "C1o", label="Data")

    mean = y["mean"]
    upper = y["mean"] + y["std"]
    lower = y["mean"] - y["std"]

    if isinstance(baseline, torch.Tensor):
        mean -= m
        upper -= m
        lower -= m
    ax.plot(mean, "C0", label="Model prediction")
    ax.fill_between(range(len(m)), int(upper), int(lower), alpha=0.25)
    ax.set_xlabel("step")
    ax.set_ylabel("B (arb. units)")
    ax.legend()

    return fig, ax
