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


import torch
import typing


def mape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor | None = None,
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
    reduction : Literal["mean", "sum"], optional
        The reduction method to apply to the loss, by default "mean". If None, the
        loss is returned as-is.

    Returns
    -------
    torch.Tensor
        The MAPE loss.
    """
    diff = torch.abs((y_true - y_pred) / y_true)
    if weights is not None:
        diff = diff * weights
    if reduction is None:
        return diff
    elif reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def smape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor | None = None,
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
    reduction : Literal["mean", "sum"], optional
        The reduction method to apply to the loss, by default "mean". If None, the
        loss is returned as-is.

    Returns
    -------
    torch.Tensor
        The SMAPE loss.
    """
    diff = (
        2
        * torch.abs(y_pred - y_true)
        / (torch.abs(y_true) + torch.abs(y_pred))
    )
    if weights is not None:
        diff = diff * weights
    if reduction is None:
        return diff
    elif reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
