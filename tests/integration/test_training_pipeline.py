"""Integration tests for full training pipeline."""

from __future__ import annotations

import pytest
import torch

from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformerModel,
)

from ..factories import TFTModelFactory, create_sample_batch
from ..test_utils import (
    TestDataGenerator,
    assert_tensor_finite,
    assert_tensor_shape,
)


@pytest.mark.integration
@pytest.mark.slow
def test_full_training_pipeline(tmp_path):
    """Integration test for full training pipeline."""
    # Create temporary data
    data = TestDataGenerator.create_hysteresis_data(length=500)
    data_path = tmp_path / "test_data.parquet"
    data.to_parquet(data_path)

    # Create model using factory
    model_config = {
        "num_past_features": 2,
        "num_future_features": 2,
        "ctxt_seq_len": 50,
        "tgt_seq_len": 25,
        "num_lstm_layers": 1,
        "n_dim_model": 32,
        "num_heads": 4,
        "output_dim": 1,
        "hidden_continuous_dim": 16,
    }

    model = TFTModelFactory.create(**model_config)

    # Test forward pass
    batch = create_sample_batch(
        batch_size=4,
        ctxt_seq_len=50,
        tgt_seq_len=25,
        num_features=2,
    )

    output = model(batch["encoder_input"], batch["decoder_input"])

    assert_tensor_shape(output["output"], (4, 25, 1))
    assert_tensor_finite(output["output"])


@pytest.mark.integration
def test_model_training_step():
    """Test that model can perform training steps."""
    model = TemporalFusionTransformerModel(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=20,
        tgt_seq_len=10,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=16,
    )

    # Create sample batch
    batch = create_sample_batch(
        batch_size=4,
        ctxt_seq_len=20,
        tgt_seq_len=10,
        num_features=2,
    )

    # Test forward pass
    output = model(batch["encoder_input"], batch["decoder_input"])

    # Simulate training step
    target = torch.randn(4, 10, 1)
    loss = torch.nn.functional.mse_loss(output["output"], target)

    # Test backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for parameter {name}"
            )


@pytest.mark.integration
def test_model_different_sequence_lengths():
    """Test model with different sequence lengths."""
    model = TemporalFusionTransformerModel(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=100,
        tgt_seq_len=50,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=16,
    )

    # Test with different sequence lengths
    test_cases = [
        (30, 15),
        (50, 25),
        (100, 50),
    ]

    for ctxt_len, tgt_len in test_cases:
        past = torch.randn(2, ctxt_len, 2)
        future = torch.randn(2, tgt_len, 2)

        output = model(past, future)

        assert output["output"].shape == (2, tgt_len, 1)
        assert torch.isfinite(output["output"]).all()


@pytest.mark.integration
def test_model_with_static_features():
    """Test model with static features (currently unused but should not break)."""
    model = TemporalFusionTransformerModel(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=30,
        tgt_seq_len=15,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=16,
        num_static_features=3,
    )

    past = torch.randn(2, 30, 2)
    future = torch.randn(2, 15, 2)
    static = torch.randn(2, 3)

    output = model(past, future, static)

    assert output["output"].shape == (2, 15, 1)
    assert torch.isfinite(output["output"]).all()
