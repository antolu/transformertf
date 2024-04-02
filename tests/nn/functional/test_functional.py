from __future__ import annotations


import typing
import torch
import pytest

from transformertf.nn.functional import mape_loss, smape_loss


@pytest.fixture
def y_pred() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def y_true() -> torch.Tensor:
    return torch.tensor([2.0, 3.0, 4.0])


@pytest.fixture
def weights() -> torch.Tensor:
    return torch.tensor([0.5, 0.25, 1.0])


@pytest.fixture(params=[None, "weights"])
def weights_or_none(request: pytest.FixtureRequest) -> torch.Tensor | None:
    if request.param == "weights":
        # get the weights fixture
        return request.getfixturevalue("weights")
    else:
        return None


@pytest.mark.parametrize(
    "y_pred, y_true, weights_or_none, reduction, expected_loss",
    [
        (y_pred, y_true, None, None, [0.5, 0.3333, 0.25]),
        (y_pred, y_true, None, "mean", 0.3611),
        (y_pred, y_true, None, "sum", 1.0833),
        (y_pred, y_true, "weights", "mean", 0.1944),
        (y_pred, y_true, "weights", "sum", 0.5833),
    ],
    indirect=["y_pred", "y_true", "weights_or_none"],
)
def test_mape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights_or_none: torch.Tensor | None,
    reduction: typing.Literal["mean", "sum"] | None,
    expected_loss: float,
) -> None:
    loss = mape_loss(y_pred, y_true, weights_or_none, reduction)
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-4)


@pytest.mark.parametrize(
    "y_pred, y_true, weights_or_none, reduction, expected_loss",
    [
        (y_pred, y_true, None, None, [0.6667, 0.4, 0.2857]),
        (y_pred, y_true, None, "mean", 0.4507),
        (y_pred, y_true, None, "sum", 1.3524),
        (y_pred, y_true, "weights", None, [0.3333, 0.1, 0.2857]),
        (y_pred, y_true, "weights", "mean", 0.2397),
        (y_pred, y_true, "weights", "sum", 0.7190),
    ],
    indirect=["y_pred", "y_true", "weights_or_none"],
)
def test_smape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights_or_none: torch.Tensor,
    reduction: typing.Literal["mean", "sum"] | None,
    expected_loss: float,
) -> None:
    loss = smape_loss(y_pred, y_true, weights_or_none, reduction)
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-4)
