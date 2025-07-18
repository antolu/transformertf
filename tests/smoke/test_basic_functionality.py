"""Smoke tests to verify basic functionality."""

from __future__ import annotations

import pytest
import torch

from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformerModel,
)


@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests to verify basic functionality."""

    def test_imports(self):
        """Test that all modules can be imported."""
        import transformertf
        import transformertf.data
        import transformertf.models
        import transformertf.nn

        assert transformertf is not None

    def test_model_creation(self):
        """Test that models can be created without errors."""
        model = TemporalFusionTransformerModel(
            num_past_features=2,
            num_future_features=2,
            ctxt_seq_len=10,
            tgt_seq_len=5,
            num_lstm_layers=1,
            n_dim_model=16,
            num_heads=2,
            output_dim=1,
            hidden_continuous_dim=8,
        )
        assert model is not None

    def test_basic_forward_pass(self):
        """Test basic forward pass works."""
        model = TemporalFusionTransformerModel(
            num_past_features=2,
            num_future_features=2,
            ctxt_seq_len=10,
            tgt_seq_len=5,
            num_lstm_layers=1,
            n_dim_model=16,
            num_heads=2,
            output_dim=1,
            hidden_continuous_dim=8,
        )

        past = torch.randn(2, 10, 2)
        future = torch.randn(2, 5, 2)

        output = model(past, future)
        assert output is not None
        assert "output" in output
        assert output["output"].shape == (2, 5, 1)

    def test_attention_module(self):
        """Test that attention module works."""
        from transformertf.nn import InterpretableMultiHeadAttention

        attention = InterpretableMultiHeadAttention(
            n_dim_model=16,
            n_heads=2,
            dropout=0.1,
        )

        input_tensor = torch.randn(2, 10, 16)
        output = attention(input_tensor, input_tensor, input_tensor)

        assert output.shape == (2, 10, 16)

    def test_quantile_loss(self):
        """Test that quantile loss works."""
        from transformertf.nn import QuantileLoss

        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        predictions = torch.randn(5, 10, 3)
        targets = torch.randn(5, 10)

        loss = loss_fn.loss(predictions, targets)

        assert loss.shape == (5, 3)  # (batch_size, num_quantiles)
        assert torch.isfinite(loss).all()

    def test_polynomial_transform(self):
        """Test that polynomial transform works."""
        from transformertf.data.transform import PolynomialTransform

        transform = PolynomialTransform(degree=2, num_iterations=10)

        input_tensor = torch.randn(5)
        output = transform(input_tensor)

        assert output.shape == (5,)
        assert torch.isfinite(output).all()

    def test_device_compatibility(self):
        """Test that models work on available devices."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TemporalFusionTransformerModel(
            num_past_features=2,
            num_future_features=2,
            ctxt_seq_len=10,
            tgt_seq_len=5,
            num_lstm_layers=1,
            n_dim_model=16,
            num_heads=2,
            output_dim=1,
            hidden_continuous_dim=8,
        ).to(device)

        past = torch.randn(2, 10, 2).to(device)
        future = torch.randn(2, 5, 2).to(device)

        output = model(past, future)
        assert output["output"].device.type == device
