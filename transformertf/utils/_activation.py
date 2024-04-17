"""
Activation function factory
"""

from __future__ import annotations

import typing

import torch

ACTIVATIONS = typing.Literal["elu", "relu", "gelu"]
ACTIVATION_MAP: dict[ACTIVATIONS, type[torch.nn.Module]] = {
    "elu": torch.nn.ELU,  # type: ignore[attr-defined]
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
}


def get_activation(
    activation: ACTIVATIONS,
    **activation_kwargs: typing.Any,
) -> torch.nn.ELU | torch.nn.ReLU | torch.nn.GELU:
    if activation not in ACTIVATION_MAP:
        msg = f"activation must be one of {list(ACTIVATION_MAP)}, not {activation}"
        raise ValueError(msg)

    return ACTIVATION_MAP[activation](**activation_kwargs)  # type: ignore[call-arg]
