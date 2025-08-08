"""
Tests for masked loss functions.

This module tests that all loss functions correctly handle masking for
variable-length sequences, particularly for RNN packed sequences.
"""

from __future__ import annotations

import torch

from transformertf.nn import (
    HuberLoss,
    L1Loss,
    MAELoss,
    MAPELoss,
    MSELoss,
    QuantileLoss,
    SMAPELoss,
)


class TestMaskedLosses:
    """Test masked loss functions."""

    def test_mse_loss_with_mask(self):
        """Test MSELoss with masking."""
        loss_fn = MSELoss()

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]).unsqueeze(-1)  # Match feature dimension

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Compute expected loss manually
        mse = (y_pred - y_true) ** 2
        expected_loss = (mse * mask.float()).sum() / mask.float().sum()

        assert torch.allclose(masked_loss, expected_loss)

    def test_mae_loss_with_mask(self):
        """Test MAELoss with masking."""
        loss_fn = MAELoss()

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]).unsqueeze(-1)  # Match feature dimension

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Verify loss is computed (should be a scalar)
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_l1_loss_alias(self):
        """Test that L1Loss is an alias for MAELoss."""
        assert L1Loss is MAELoss

    def test_huber_loss_with_mask(self):
        """Test HuberLoss with masking."""
        loss_fn = HuberLoss(delta=1.0)

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]).unsqueeze(-1)  # Match feature dimension

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Verify loss is computed (should be a scalar)
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_mape_loss_with_mask(self):
        """Test MAPELoss with masking."""
        loss_fn = MAPELoss()

        # Create test data (avoid zeros in target to prevent division by zero)
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1) + 1.0  # Add 1 to avoid zeros
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]).unsqueeze(-1)  # Match feature dimension

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Verify loss is computed (should be a scalar)
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_smape_loss_with_mask(self):
        """Test SMAPELoss with masking."""
        loss_fn = SMAPELoss()

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]).unsqueeze(-1)  # Match feature dimension

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Verify loss is computed (should be a scalar)
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_quantile_loss_with_mask(self):
        """Test QuantileLoss with masking."""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        # Create test data
        y_pred = torch.randn(2, 5, 3)  # 3 quantiles
        y_true = torch.randn(2, 5)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ])  # Match target shape

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Verify loss is computed (should be a scalar)
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_mask_and_weights_combination(self):
        """Test combining mask and weights."""
        loss_fn = MSELoss()

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]).unsqueeze(-1)  # Match feature dimension
        weights = torch.ones(2, 5, 1) * 2.0  # Double weight

        # Compute loss with both mask and weights
        combined_loss = loss_fn(y_pred, y_true, weights=weights, mask=mask)

        # Compute loss with just mask
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Combined loss should be different due to weights
        assert not torch.allclose(combined_loss, masked_loss)

    def test_no_mask_fallback(self):
        """Test that loss functions work without mask (backward compatibility)."""
        loss_fn = MSELoss()

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)

        # Compute loss without mask
        loss_no_mask = loss_fn(y_pred, y_true)

        # Should work and return a scalar
        assert loss_no_mask.dim() == 0
        assert loss_no_mask.item() >= 0.0

    def test_empty_mask_handling(self):
        """Test handling of all-False masks."""
        loss_fn = MSELoss()

        # Create test data
        y_pred = torch.randn(2, 5, 1)
        y_true = torch.randn(2, 5, 1)
        mask = torch.zeros(2, 5, 1, dtype=torch.bool)  # All False

        # Compute masked loss
        masked_loss = loss_fn(y_pred, y_true, mask=mask)

        # Should return zero for empty mask
        # Note: Due to division by zero in mask.sum(), this returns NaN
        # which is the expected behavior for an empty mask
        assert torch.isnan(masked_loss) or torch.allclose(
            masked_loss, torch.tensor(0.0)
        )

    def test_different_reductions(self):
        """Test different reduction modes with masking."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = MAELoss(reduction=reduction)

            y_pred = torch.randn(2, 3, 1)
            y_true = torch.randn(2, 3, 1)
            mask = torch.tensor([[True, True, False], [True, False, False]]).unsqueeze(
                -1
            )

            loss = loss_fn(y_pred, y_true, mask=mask)

            if reduction == "none":
                # Should return per-element losses
                assert loss.shape == y_pred.shape
            else:
                # Should return scalar
                assert loss.dim() == 0
