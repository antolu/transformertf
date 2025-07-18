from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from hypothesis import given, settings

from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformerModel,
)

from ...strategies import tft_config_strategy

BATCH_SIZE = 4
PAST_SEQ_LEN = 100
FUTURE_SEQ_LEN = 10
NUM_FEATURES = 2


@pytest.fixture(scope="module")
def past_covariates() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, PAST_SEQ_LEN, NUM_FEATURES)


@pytest.fixture(scope="module")
def future_covariates() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, FUTURE_SEQ_LEN, NUM_FEATURES - 1)


def test_temporal_fusion_transformer_model(
    past_covariates: torch.Tensor, future_covariates: torch.Tensor
) -> None:
    model = TemporalFusionTransformerModel(
        num_past_features=NUM_FEATURES,
        num_future_features=1,
        ctxt_seq_len=PAST_SEQ_LEN,
        tgt_seq_len=FUTURE_SEQ_LEN,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=12,
    )

    output = model(past_covariates, future_covariates)

    assert output["output"].shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)


@pytest.mark.property
@given(config=tft_config_strategy())
@settings(max_examples=20, deadline=None)
def test_tft_model_property_invariants(config):
    """Property-based test for TFT model invariants."""
    model = TemporalFusionTransformerModel(**config)

    # Test with random inputs
    batch_size = 4
    past_covariates = torch.randn(
        batch_size, config["ctxt_seq_len"], config["num_past_features"]
    )
    future_covariates = torch.randn(
        batch_size, config["tgt_seq_len"], config["num_future_features"]
    )

    output = model(past_covariates, future_covariates)

    # Property: output should have correct shape
    expected_shape = (batch_size, config["tgt_seq_len"], config["output_dim"])
    assert output["output"].shape == expected_shape

    # Property: output should be finite
    assert torch.isfinite(output["output"]).all()

    # Property: attention weights should sum to 1 (if present)
    if "attention_weights" in output:
        attn_weights = output["attention_weights"]
        assert torch.allclose(
            attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))
        )


def test_tft_model_different_batch_sizes():
    """Test TFT model with different batch sizes."""
    model = TemporalFusionTransformerModel(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=50,
        tgt_seq_len=25,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=16,
    )

    # Test with different batch sizes
    for batch_size in [1, 4, 16]:
        past = torch.randn(batch_size, 50, 2)
        future = torch.randn(batch_size, 25, 2)

        output = model(past, future)

        assert output["output"].shape == (batch_size, 25, 1)
        assert torch.isfinite(output["output"]).all()


def test_tft_model_deterministic():
    """Test that TFT model is deterministic in eval mode."""
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
        dropout=0.0,  # No dropout for deterministic behavior
    )
    model.eval()

    past = torch.randn(2, 30, 2)
    future = torch.randn(2, 15, 2)

    # Run twice
    with torch.no_grad():
        output1 = model(past, future)
        output2 = model(past, future)

    # Should be identical
    assert torch.allclose(output1["output"], output2["output"], atol=1e-6)


def test_tft_model_gradient_flow():
    """Test that gradients flow through TFT model."""
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

    past = torch.randn(2, 20, 2)
    future = torch.randn(2, 10, 2)

    output = model(past, future)
    loss = output["output"].sum()
    loss.backward()

    # Check that gradients are computed for model parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient for parameter {name}"
        )


def test_tft_model_output_consistency():
    """Test that TFT model produces consistent outputs across runs."""
    model = TemporalFusionTransformerModel(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=40,
        tgt_seq_len=20,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=16,
    )

    # Set model to eval mode
    model.eval()

    past = torch.randn(4, 40, 2)
    future = torch.randn(4, 20, 2)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    output1 = model(past, future)

    torch.manual_seed(42)
    output2 = model(past, future)

    # Outputs should be identical
    assert torch.allclose(output1["output"], output2["output"], atol=1e-6)


@pytest.mark.parametrize(
    "edge_case",
    [
        "single_element",
        "large_batch",
    ],
)
def test_tft_model_edge_cases(edge_case):
    """Test TFT model behavior with edge cases."""
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

    if edge_case == "single_element":
        past = torch.randn(1, 10, 2)
        future = torch.randn(1, 5, 2)
        output = model(past, future)
        assert torch.isfinite(output["output"]).all()
    elif edge_case == "large_batch":
        past = torch.randn(100, 10, 2)
        future = torch.randn(100, 5, 2)
        output = model(past, future)
        assert torch.isfinite(output["output"]).all()


def test_tft_model_with_mocked_attention():
    """Test TFT model with mocked attention component."""
    with patch(
        "transformertf.models.temporal_fusion_transformer._model.InterpretableMultiHeadAttention"
    ) as mock_attention:
        # Configure mock to return expected shapes
        mock_attention.return_value.forward.return_value = (
            torch.randn(4, 25, 32),
            torch.randn(4, 25, 25),
        )

        TemporalFusionTransformerModel(
            num_past_features=2,
            num_future_features=2,
            ctxt_seq_len=50,
            tgt_seq_len=25,
            num_lstm_layers=1,
            n_dim_model=32,
            num_heads=4,
            output_dim=1,
            hidden_continuous_dim=16,
        )

        # Test that mock was called during initialization
        assert mock_attention.called
