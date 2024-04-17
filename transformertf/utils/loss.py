from __future__ import annotations

import typing

import torch

from ..nn import QuantileLoss, WeightedHuberLoss, WeightedMAELoss, WeightedMSELoss

VALID_LOSS = typing.Literal["mse", "mae", "quantile", "huber"]


LOSS_MAP: dict[VALID_LOSS, type[torch.nn.Module]] = {
    "huber": WeightedHuberLoss,  # "huber" is an alias for "smooth_l1
    "mse": WeightedMSELoss,
    "mae": WeightedMAELoss,
    "quantile": QuantileLoss,
}


def get_loss(
    loss: VALID_LOSS,
    **loss_kwargs: typing.Any,
) -> torch.nn.MSELoss | QuantileLoss | torch.nn.L1Loss | torch.nn.HuberLoss:
    if loss not in LOSS_MAP:
        valid_losses = list(LOSS_MAP)
        error_message = f"loss must be one of {valid_losses}, not {loss}"
        raise ValueError(error_message)

    return LOSS_MAP[loss](**loss_kwargs)  # type: ignore[call-arg,return-value]
