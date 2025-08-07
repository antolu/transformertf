from __future__ import annotations

import typing

import pytest
import torch

import transformertf.nn


@pytest.fixture
def y_pred() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def target() -> torch.Tensor:
    return torch.tensor([1.0, 3.0, 4.0])


@pytest.fixture
def weights() -> torch.Tensor:
    return torch.tensor([0.5, 1.0, 1.5])


@pytest.fixture(params=[None, "weights"])
def weights_or_none(request: pytest.FixtureRequest) -> torch.Tensor | None:
    if request.param == "weights":
        # get the weights fixture
        return request.getfixturevalue("weights")

    return None


@pytest.mark.parametrize(
    ("y_pred", "target", "weights_or_none", "reduction", "expected"),
    [
        (y_pred, target, None, "mean", torch.tensor(2.0 / 3)),
        (y_pred, target, None, "sum", torch.tensor(2.0)),
        (y_pred, target, None, "none", torch.tensor([0.0, 1.0, 1.0])),
        (y_pred, target, "weights", "mean", torch.tensor(5.0 / 6)),
        (y_pred, target, "weights", "sum", torch.tensor(2.5)),
        (y_pred, target, "weights", "none", torch.tensor([0.0, 1.0, 1.5])),
    ],
    indirect=["y_pred", "target", "weights_or_none"],
)
def test_weighted_mse(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    weights_or_none: torch.Tensor,
    reduction: typing.Literal["mean", "sum", "none"],
    expected: torch.Tensor,
) -> None:
    loss_fn = transformertf.nn.MSELoss(reduction=reduction)
    loss = loss_fn(y_pred, target, weights=weights_or_none)
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    ("y_pred", "target", "weights_or_none", "reduction", "expected"),
    [
        (y_pred, target, None, "mean", torch.tensor(2.0 / 3)),
        (y_pred, target, None, "sum", torch.tensor(2.0)),
        (y_pred, target, None, "none", torch.tensor([0.0, 1.0, 1.0])),
        (y_pred, target, "weights", "mean", torch.tensor(5.0 / 6)),
        (y_pred, target, "weights", "sum", torch.tensor(2.5)),
        (y_pred, target, "weights", "none", torch.tensor([0.0, 1.0, 1.5])),
    ],
    indirect=["y_pred", "target", "weights_or_none"],
)
def test_weighted_mae(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    weights_or_none: torch.Tensor,
    reduction: typing.Literal["mean", "sum", "none"],
    expected: torch.Tensor,
) -> None:
    loss_fn = transformertf.nn.MAELoss(reduction=reduction)
    loss = loss_fn(y_pred, target, weights=weights_or_none)
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    ("y_pred", "target", "weights_or_none", "reduction", "expected"),
    [
        (y_pred, target, None, "mean", torch.tensor(1.0 / 3)),
        (y_pred, target, None, "sum", torch.tensor(1.0)),
        (y_pred, target, None, "none", torch.tensor([0.0, 0.5, 0.5])),
        (y_pred, target, "weights", "mean", torch.tensor(5.0 / 12)),
        (y_pred, target, "weights", "sum", torch.tensor(1.25)),
        (y_pred, target, "weights", "none", torch.tensor([0.0, 0.5, 0.75])),
    ],
    indirect=["y_pred", "target", "weights_or_none"],
)
def test_weighted_huber(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    weights_or_none: torch.Tensor,
    reduction: typing.Literal["mean", "sum", "none"],
    expected: torch.Tensor,
) -> None:
    loss_fn = transformertf.nn.HuberLoss(reduction=reduction)
    loss = loss_fn(y_pred, target, weights=weights_or_none)
    assert torch.allclose(loss, expected)
