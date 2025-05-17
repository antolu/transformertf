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
    if mask is None and weight is None:
        return torch.nn.functional.mse_loss(input, target, reduction=reduction)

    target = target.view_as(input)
    loss = torch.nn.functional.mse_loss(input, target, reduction="none")
    loss = (loss * mask) if mask is not None else loss

    if weight is not None:
        while weight.dim() < loss.dim():
            weight = weight.unsqueeze(-1)
        loss *= weight

    if regularization is not None:
        loss += regularization * torch.gradient(loss, dim=regularization_dim)[0]

    if reduction == "mean":
        if mask is None:
            return loss.mean()
        return loss.sum() / mask.sum()
    if reduction == "sum":
        return torch.nansum(loss)
    return loss
