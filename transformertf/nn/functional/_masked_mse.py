from __future__ import annotations

import typing

import torch

from ...utils import maybe_compile


@maybe_compile
def mse_loss(
    input: torch.Tensor,  # noqa: A002
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    regularization: float | None = None,
    regularization_dim: int = 0,
    regularization_order: typing.Literal[1, 2, 3] = 1,
    reduction: typing.Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute the mean squared error loss with optional masking and weighting, and
    optional regularization.

    The regularization term is the gradient of the loss multiplied by a scalar, to
    smooth the prediction.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
    weight : torch.Tensor, optional
        The weight tensor, by default None. The weight tensor should have the same shape as
        the input tensor. If None, no weighting is applied.
    mask : torch.Tensor, optional
        The mask tensor, by default None. The mask tensor should have the same shape as
        the input tensor. If None, no masking is applied.
    regularization : float, optional
        The regularization term, by default None. If None, no regularization is applied.
    reduction : str, optional
        The reduction method, by default "mean". Can be "mean", "sum", or "none".
        - "mean": the sum of the output will be divided by the number of elements in the
            output.
        - "sum": the output will be summed.
        - "none": no reduction will be applied, and the output will have the same shape as
            the input.
    """
    target = target.view_as(input)

    if mask is None and weight is None:
        return torch.nn.functional.mse_loss(input, target, reduction=reduction)

    loss = torch.nn.functional.mse_loss(input, target, reduction="none")
    loss = (loss * mask) if mask is not None else loss

    if weight is not None:
        while weight.dim() < loss.dim():
            weight = weight.unsqueeze(-1)
        loss *= weight

    if regularization is not None:
        loss += regularization * manual_smoothness(
            loss,
            order=regularization_order,
            reduction="none",
            dim=regularization_dim,
        ).sum(dim=regularization_dim)

    if reduction == "mean":
        if mask is None:
            return loss.mean()
        return loss.sum() / mask.sum()
    if reduction == "sum":
        return torch.nansum(loss)
    return loss


def manual_smoothness(
    error: torch.Tensor,
    order: typing.Literal[1, 2, 3],
    reduction: typing.Literal["mean", "sum", "none"] | None = "mean",
    dim: int = 0,
) -> torch.Tensor:
    # Create slice objects for the specified dimension
    def slice_dim(start: int | None = None, stop: int | None = None) -> list:
        slices = [slice(None)] * error.dim()
        slices[dim] = slice(start, stop)
        return slices

    if order == 1:
        diff = error[slice_dim(1, None)] - error[slice_dim(None, -1)]
    elif order == 2:
        diff = (
            error[slice_dim(2, None)]
            - 2 * error[slice_dim(1, -1)]
            + error[slice_dim(None, -2)]
        )
    elif order == 3:
        diff = (
            error[slice_dim(3, None)]
            - 3 * error[slice_dim(2, -1)]
            + 3 * error[slice_dim(1, -2)]
            - error[slice_dim(None, -3)]
        )
    else:
        msg = f"Unsupported order: {order}. Supported orders are 1, 2, and 3."
        raise ValueError(msg)

    if reduction == "mean":
        return torch.mean(diff**2)
    if reduction == "sum":
        return torch.sum(diff**2)
    if reduction == "none" or reduction is None:
        return diff**2

    msg = f"Unsupported reduction: {reduction}. Supported reductions are 'mean', 'sum', and 'none'."
    raise ValueError(msg)
