from __future__ import annotations

import pytest
import torch

from transformertf.nn import get_activation


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("elu", torch.nn.ELU),
        ("relu", torch.nn.ReLU),
        ("gelu", torch.nn.GELU),
    ],
)
def test_get_activation(activation: str, expected: type[torch.nn.Module]) -> None:
    activation_fn = get_activation(activation)  # type: ignore[arg-type]
    assert isinstance(activation_fn, expected)


def test_get_unknown_activation() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        get_activation("something")  # type: ignore[arg-type]
