from __future__ import annotations

import typing

import torch

from ...utils import maybe_compile


@maybe_compile
def masked_mse_loss(
    input: torch.Tensor,  # noqa: A002
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    reduction: typing.Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    if mask is None and weight is None:
        return torch.nn.functional.mse_loss(input, target, reduction=reduction)

    target = target.view_as(input)
    loss = torch.nn.functional.mse_loss(input, target, reduction="none")
    loss = (loss * mask) if mask is not None else loss

    if weight is not None:
        while weight.dim() < loss.dim():
            weight = weight.unsqueeze(-1)
        loss *= weight

    if reduction == "mean":
        if mask is None:
            return loss.mean()
        return loss.sum() / mask.sum()
    if reduction == "sum":
        return torch.nansum(loss)
    return loss
