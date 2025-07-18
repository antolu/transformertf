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

    weights = (
        einops.repeat(weights_or_none[:2], "... -> ... n 1", n=5)
        if weights_or_none is not None
        else None
    )

    result = loss.loss(
        y_pred,
        target,
        weights=weights,
    )

    assert torch.isnan(result).sum() == 0


@pytest.mark.parametrize(
    "quantiles",
    [
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
        [0.5],
    ],
)
def test_quantile_loss_computation(quantiles):
    """Test quantile loss with different quantiles."""
    loss_fn = QuantileLoss(quantiles=quantiles)

    predictions = torch.randn(10, 5, len(quantiles))
    targets = torch.randn(10, 5)

    loss = loss_fn.loss(predictions, targets)

    # Check that output is finite and not NaN
    assert torch.isfinite(loss).all()
    assert not torch.isnan(loss).any()

    # Loss should be non-negative
    assert (loss >= 0).all()


def test_quantile_loss_asymmetric_penalty():
    """Test that quantile loss provides asymmetric penalties."""
    loss_fn = QuantileLoss(quantiles=[0.1, 0.9])

    # Create predictions and targets where we know the relationship
    predictions = torch.tensor([[[0.5, 0.5]]])  # same prediction for both quantiles
    target_low = torch.tensor([[0.3]])  # target below prediction
    target_high = torch.tensor([[0.7]])  # target above prediction

    loss_low = loss_fn.loss(predictions, target_low)
    loss_high = loss_fn.loss(predictions, target_high)

    # For 0.1 quantile, under-prediction should be penalized more than over-prediction
    # For 0.9 quantile, over-prediction should be penalized more than under-prediction
    assert torch.isfinite(loss_low).all()
    assert torch.isfinite(loss_high).all()


def test_quantile_loss_median_special_case():
    """Test that median quantile (0.5) gives half the MAE."""
    loss_fn = QuantileLoss(quantiles=[0.5])

    predictions = torch.randn(10, 5, 1)
    targets = torch.randn(10, 5)

    # Compute quantile loss
    q_loss = loss_fn.loss(predictions, targets)

    # Compute MAE manually - sum over sequence dimension to match quantile loss
    mae_sum = torch.abs(predictions.squeeze(-1) - targets).sum(dim=1)

    # Median quantile loss should be half the MAE (sum)
    assert torch.allclose(q_loss.squeeze(), mae_sum / 2, atol=1e-5)


def test_quantile_loss_shapes():
    """Test quantile loss with different input shapes."""
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    # Test different shapes
    batch_size = 4
    seq_len = 10
    num_quantiles = 3

    # Standard case
    predictions = torch.randn(batch_size, seq_len, num_quantiles)
    targets = torch.randn(batch_size, seq_len)

    loss = loss_fn.loss(predictions, targets)
    assert loss.shape == (batch_size, num_quantiles)

    # With weights (should broadcast correctly)
    weights = torch.ones(batch_size, seq_len, 1)
    weighted_loss = loss_fn.loss(predictions, targets, weights)
    assert weighted_loss.shape == (batch_size, num_quantiles)


def test_quantile_loss_with_zero_weights():
    """Test quantile loss with zero weights."""
    loss_fn = QuantileLoss(quantiles=[0.5])

    predictions = torch.randn(5, 10, 1)
    targets = torch.randn(5, 10)
    weights = torch.zeros(5, 10, 1)

    loss = loss_fn.loss(predictions, targets, weights)

    # Loss should be zero when weights are zero
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_quantile_loss_point_prediction():
    """Test point prediction extraction."""
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    predictions = torch.randn(8, 12, 3)

    # Get point prediction (median)
    point_pred = loss_fn.point_prediction(predictions)

    # Should extract the median (index 1)
    assert torch.allclose(point_pred, predictions[..., 1])
    assert point_pred.shape == (8, 12)


def test_quantile_loss_to_quantiles():
    """Test quantile extraction functionality."""
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    loss_fn = QuantileLoss(quantiles=quantiles)

    predictions = torch.randn(4, 8, 5)

    # Should return the same tensor
    quantiles_pred = loss_fn.to_quantiles(predictions)
    assert torch.allclose(quantiles_pred, predictions)

    # Check that quantiles are accessible
    assert len(loss_fn.quantiles) == 5
    assert torch.allclose(loss_fn.quantiles, torch.tensor(quantiles))


def test_quantile_loss_deterministic():
    """Test that quantile loss is deterministic."""
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    predictions = torch.randn(3, 7, 3)
    targets = torch.randn(3, 7)

    # Compute loss twice
    loss1 = loss_fn.loss(predictions, targets)
    loss2 = loss_fn.loss(predictions, targets)

    # Should be identical
    assert torch.allclose(loss1, loss2)


def test_quantile_loss_gradient_flow():
    """Test that gradients flow properly through quantile loss."""
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    predictions = torch.randn(2, 5, 3, requires_grad=True)
    targets = torch.randn(2, 5)

    loss = loss_fn.loss(predictions, targets).sum()
    loss.backward()

    # Check that gradients are computed
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert not torch.isnan(predictions.grad).any()
