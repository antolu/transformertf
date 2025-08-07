"""
This module contains the functional interface for some functions, equivalent to the
torch.nn.functional module.

Currently, the module contains functions to calculate

- MAPE
- SMAPE
- MASE

with weights and reduction options.
"""

from __future__ import annotations

import typing

import torch


def mape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    reduction: typing.Literal["mean", "sum"] | None = "mean",
) -> torch.Tensor:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between the predicted and true
    values.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted values.
    y_true : torch.Tensor
        The true values.
    weights : torch.Tensor, optional
        The weights to apply to the loss, by default None.
    mask : torch.Tensor, optional
        Numeric mask to exclude certain positions from loss calculation, by default None.
        Values of 1.0 include the position, 0.0 excludes it. Intermediate values provide partial weighting.
    reduction : Literal["mean", "sum"], optional
        The reduction method to apply to the loss, by default "mean". If None, the
        loss is returned as-is.

    Returns
    -------
    torch.Tensor
        The MAPE loss.
    """
    diff = torch.abs((y_true - y_pred) / y_true)

    # Combine mask and weights
    if mask is not None and weights is not None:
        diff *= weights * mask.float()
    elif mask is not None:
        diff *= mask.float()
    elif weights is not None:
        diff *= weights
    if reduction is None:
        return diff
    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    msg = f"Invalid reduction method: {reduction}"
    raise ValueError(msg)


def smape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    reduction: typing.Literal["mean", "sum"] | None = "mean",
) -> torch.Tensor:
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between the predicted
    and true values.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted values.
    y_true : torch.Tensor
        The true values.
    weights : torch.Tensor, optional
        The weights to apply to the loss, by default None.
    mask : torch.Tensor, optional
        Numeric mask to exclude certain positions from loss calculation, by default None.
        Values of 1.0 include the position, 0.0 excludes it. Intermediate values provide partial weighting.
    reduction : Literal["mean", "sum"], optional
        The reduction method to apply to the loss, by default "mean". If None, the
        loss is returned as-is.

    Returns
    -------
    torch.Tensor
        The SMAPE loss.
    """
    diff = 2 * torch.abs(y_pred - y_true) / (torch.abs(y_true) + torch.abs(y_pred))

    # Combine mask and weights
    if mask is not None and weights is not None:
        diff *= weights * mask.float()
    elif mask is not None:
        diff *= mask.float()
    elif weights is not None:
        diff *= weights
    if reduction is None:
        return diff
    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    msg = f"Invalid reduction method: {reduction}"
    raise ValueError(msg)
