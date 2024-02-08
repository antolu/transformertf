from __future__ import annotations

import typing

import torch

from ..nn import QuantileLoss

VALID_LOSS = typing.Literal["mse", "quantile", "huber"]


LOSS_MAP: dict[VALID_LOSS, typing.Type[torch.nn.Module]] = {
    "huber": torch.nn.HuberLoss,  # "huber" is an alias for "smooth_l1
    "mse": torch.nn.MSELoss,
    "quantile": QuantileLoss,
}


def get_loss(
    loss: VALID_LOSS,
    **loss_kwargs: typing.Any,
) -> torch.nn.MSELoss | QuantileLoss:
    if loss not in LOSS_MAP:
        raise ValueError(f"loss must be one of {list(LOSS_MAP)}, not {loss}")

    return LOSS_MAP[loss](**loss_kwargs)  # type: ignore[call-arg]
