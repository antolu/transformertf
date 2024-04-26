"""
Activation function factory
"""

from __future__ import annotations

import typing

import torch

VALID_ACTIVATIONS = typing.Literal["elu", "relu", "gelu"]
_ACTIVATION_MAP: dict[VALID_ACTIVATIONS, type[torch.nn.Module]] = {
    "elu": torch.nn.ELU,  # type: ignore[attr-defined]
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
}


def get_activation(
    activation: VALID_ACTIVATIONS,
    **activation_kwargs: typing.Any,
) -> torch.nn.ELU | torch.nn.ReLU | torch.nn.GELU:
    if activation not in _ACTIVATION_MAP:
        msg = f"activation must be one of {list(_ACTIVATION_MAP)}, not {activation}"
        raise ValueError(msg)

    return _ACTIVATION_MAP[activation](**activation_kwargs)  # type: ignore[call-arg]
