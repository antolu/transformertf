from __future__ import annotations

import einops
import pytest
import torch

from transformertf.nn import QuantileLoss


@pytest.fixture
def y_pred() -> torch.Tensor:
    return einops.repeat(
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])[None, :, None], "... 1 -> ... n", n=7
    )


@pytest.fixture
def target() -> torch.Tensor:
    return torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])[None, :]


@pytest.fixture
def weights() -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 0.75, 0.5, 1.0])


@pytest.fixture
def weights_or_none(request: pytest.FixtureRequest) -> torch.Tensor | None:
    return (
        request.getfixturevalue(request.param) if request.param == "weights" else None
    )


def test_create_quantile_loss() -> None:
    loss = QuantileLoss()

    assert torch.allclose(
        loss.quantiles, torch.tensor([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    )


def test_create_quantile_loss_with_custom_quantiles() -> None:
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    assert torch.allclose(loss.quantiles, torch.tensor([0.1, 0.5, 0.9]))


@pytest.mark.parametrize("weights_or_none", ["weights", None], indirect=True)
def test_quantile_loss(
    y_pred: torch.Tensor, target: torch.Tensor, weights_or_none: torch.Tensor | None
) -> None:
    loss = QuantileLoss(quantiles=[0.5])

    result = loss.loss(
        y_pred,
        target,
        weights=weights_or_none[0:1] if weights_or_none is not None else None,
    )

    assert torch.isnan(result).sum() == 0


@pytest.mark.parametrize("weights_or_none", ["weights", None], indirect=True)
def test_quantile_loss_batch(
    y_pred: torch.Tensor, target: torch.Tensor, weights_or_none: torch.Tensor | None
) -> None:
    loss = QuantileLoss()

    y_pred = torch.cat([y_pred, y_pred])
    target = torch.cat([target, target])

    result = loss.loss(
        y_pred,
        target,
        weights=weights_or_none[:2] if weights_or_none is not None else None,
    )

    assert torch.isnan(result).sum() == 0
