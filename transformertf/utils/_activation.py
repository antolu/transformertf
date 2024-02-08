"""
Activation function factory
"""
from __future__ import annotations

import typing

import torch

ACTIVATIONS = typing.Literal["relu", "gelu"]
ACTIVATION_MAP: dict[ACTIVATIONS, typing.Type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
}


def get_activation(
    activation: typing.Literal["relu", "gelu"],
    **actiation_kwargs: typing.Any,
) -> torch.nn.ReLU | torch.nn.GELU:
    if activation not in ACTIVATION_MAP:
        raise ValueError(
            f"activation must be one of {list(ACTIVATION_MAP)}, not {activation}"
        )

    return ACTIVATION_MAP[activation](**actiation_kwargs)  # type: ignore[call-arg]
