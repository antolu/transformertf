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

    _loss = torch.nn.functional.mse_loss(input, target, reduction="none")
    _loss = (_loss * mask) if mask is not None else _loss

    if weight is not None:
        while weight.dim() < _loss.dim():
            weight = weight.unsqueeze(-1)
        _loss *= weight

    if reduction == "mean":
        if mask is None:
            return _loss.mean()
        return _loss.sum() / mask.sum()
    if reduction == "sum":
        return torch.nansum(_loss)
    return _loss
