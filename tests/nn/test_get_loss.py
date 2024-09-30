from __future__ import annotations

import pytest
import torch

from transformertf.nn import QuantileLoss, WeightedMSELoss, get_loss


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        ("mse", WeightedMSELoss),
        ("mae", torch.nn.L1Loss),
        ("quantile", QuantileLoss),
        ("huber", torch.nn.HuberLoss),
    ],
)
def test_get_loss(loss: str, expected: type[torch.nn.Module]) -> None:
    actual = get_loss(loss)  # type: ignore[arg-type]
    assert isinstance(actual, expected)


def test_get_loss_unknown() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        get_loss("unknown")  # type: ignore[arg-type]
