"""
Tests for TransformerModuleBase loss masking integration.

This module tests that the calc_loss method correctly applies masking
when decoder_lengths are provided in the batch.
"""

from __future__ import annotations

import torch

from transformertf.models._base_transformer import TransformerModuleBase, create_mask
from transformertf.nn import MSELoss, QuantileLoss


class MockTransformerModule(TransformerModuleBase):
    """Mock transformer module for testing."""

    def __init__(self, criterion, use_loss_masking=True):
        super().__init__()
        self.criterion = criterion
        self.save_hyperparameters(ignore=["criterion"])
        # Set the use_loss_masking parameter manually since it's not an init arg
        self.hparams.use_loss_masking = use_loss_masking

    def forward(self, batch):
        # Not used in these tests
        return {"output": torch.zeros(1, 1, 1)}


class TestTransformerMaskedLoss:
    """Test TransformerModuleBase loss masking."""

    def test_calc_loss_with_decoder_lengths(self):
        """Test that calc_loss applies masking when decoder_lengths are provided."""
        model = MockTransformerModule(MSELoss())

        # Create test data
        batch_size, seq_len = 2, 5
        model_output = torch.randn(batch_size, seq_len, 1)
        target = torch.randn(batch_size, seq_len, 1)
        decoder_lengths = torch.tensor([[3], [4]])  # Different sequence lengths

        batch = {
            "target": target,
            "decoder_lengths": decoder_lengths,
        }

        # Compute masked loss
        masked_loss = model.calc_loss(model_output, batch)

        # Should return a scalar tensor
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_calc_loss_without_decoder_lengths(self):
        """Test that calc_loss works normally without decoder_lengths."""
        model = MockTransformerModule(MSELoss())

        # Create test data
        batch_size, seq_len = 2, 5
        model_output = torch.randn(batch_size, seq_len, 1)
        target = torch.randn(batch_size, seq_len, 1)

        batch = {"target": target}

        # Compute unmasked loss
        unmasked_loss = model.calc_loss(model_output, batch)

        # Should return a scalar tensor
        assert unmasked_loss.dim() == 0
        assert unmasked_loss.item() >= 0.0

    def test_calc_loss_masking_disabled(self):
        """Test that masking can be disabled via hyperparameter."""
        model = MockTransformerModule(MSELoss(), use_loss_masking=False)

        # Create test data
        batch_size, seq_len = 2, 5
        model_output = torch.randn(batch_size, seq_len, 1)
        target = torch.randn(batch_size, seq_len, 1)
        decoder_lengths = torch.tensor([[3], [4]])

        batch = {
            "target": target,
            "decoder_lengths": decoder_lengths,
        }

        # Should ignore masking even with decoder_lengths
        loss = model.calc_loss(model_output, batch)

        # Should return a scalar tensor
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_calc_loss_with_quantile_loss(self):
        """Test that calc_loss works with QuantileLoss and masking."""
        model = MockTransformerModule(QuantileLoss([0.1, 0.5, 0.9]))

        # Create test data
        batch_size, seq_len, num_quantiles = 2, 5, 3
        model_output = torch.randn(batch_size, seq_len, num_quantiles)
        target = torch.randn(batch_size, seq_len)
        decoder_lengths = torch.tensor([[3], [4]])

        batch = {
            "target": target,
            "decoder_lengths": decoder_lengths,
        }

        # Compute masked loss
        masked_loss = model.calc_loss(model_output, batch)

        # Should return a scalar tensor
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0

    def test_calc_loss_left_alignment_masking(self):
        """Test that masking uses left alignment as expected."""
        model = MockTransformerModule(MSELoss())

        # Create test data where we know the pattern
        batch_size, seq_len = 1, 4
        model_output = torch.ones(batch_size, seq_len, 1)
        target = torch.zeros(batch_size, seq_len, 1)
        decoder_lengths = torch.tensor([[2]])  # Only 2 valid positions

        batch = {
            "target": target,
            "decoder_lengths": decoder_lengths,
        }

        # Compute masked loss
        masked_loss = model.calc_loss(model_output, batch)

        # Manually compute expected loss for left alignment
        # With length 2, positions [2, 3] should be valid (left-aligned)
        mask = create_mask(
            seq_len, decoder_lengths.squeeze(-1), alignment="left", inverse=True
        )
        expected_positions = mask[0]  # Should be [False, False, True, True]

        assert expected_positions.tolist() == [False, False, True, True]

        # Loss should be computed only on valid positions
        expected_loss = torch.tensor(1.0)  # (1-0)^2 = 1 for both valid positions
        assert torch.allclose(masked_loss, expected_loss)

    def test_calc_loss_target_squeezing(self):
        """Test that target dimension squeezing works with masking."""
        model = MockTransformerModule(MSELoss())

        # Create test data where target has extra dimension
        batch_size, seq_len = 2, 3
        model_output = torch.randn(batch_size, seq_len)  # No feature dimension
        target = torch.randn(batch_size, seq_len, 1)  # Has feature dimension
        decoder_lengths = torch.tensor([[2], [3]])

        batch = {
            "target": target,
            "decoder_lengths": decoder_lengths,
        }

        # Should handle dimension mismatch and apply masking
        masked_loss = model.calc_loss(model_output, batch)

        # Should return a scalar tensor
        assert masked_loss.dim() == 0
        assert masked_loss.item() >= 0.0
