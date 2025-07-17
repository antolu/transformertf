from __future__ import annotations

import pytest
import torch

from transformertf.models.attention_lstm import AttentionLSTMModel

BATCH_SIZE = 4
PAST_SEQ_LEN = 50
FUTURE_SEQ_LEN = 10
NUM_PAST_FEATURES = 5
NUM_FUTURE_FEATURES = 3


@pytest.fixture(scope="module")
def past_covariates() -> torch.Tensor:
    """Create sample past covariates tensor."""
    return torch.rand(BATCH_SIZE, PAST_SEQ_LEN, NUM_PAST_FEATURES)


@pytest.fixture(scope="module")
def future_covariates() -> torch.Tensor:
    """Create sample future covariates tensor."""
    return torch.rand(BATCH_SIZE, FUTURE_SEQ_LEN, NUM_FUTURE_FEATURES)


@pytest.fixture(scope="module")
def encoder_lengths() -> torch.Tensor:
    """Create sample encoder lengths tensor."""
    return torch.tensor([PAST_SEQ_LEN] * BATCH_SIZE)


@pytest.fixture(scope="module")
def decoder_lengths() -> torch.Tensor:
    """Create sample decoder lengths tensor."""
    return torch.tensor([FUTURE_SEQ_LEN] * BATCH_SIZE)


def test_attention_lstm_forward_pass(
    past_covariates: torch.Tensor,
    future_covariates: torch.Tensor,
    encoder_lengths: torch.Tensor,
    decoder_lengths: torch.Tensor,
) -> None:
    """Test basic forward pass of AttentionLSTM model."""
    model = AttentionLSTMModel(
        num_past_features=NUM_PAST_FEATURES,
        num_future_features=NUM_FUTURE_FEATURES,
        hidden_size=64,
        num_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    output = model(past_covariates, future_covariates, decoder_lengths)

    assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)
    assert torch.isfinite(output).all()


def test_attention_lstm_without_gating(
    past_covariates: torch.Tensor,
    future_covariates: torch.Tensor,
    encoder_lengths: torch.Tensor,
    decoder_lengths: torch.Tensor,
) -> None:
    """Test AttentionLSTM model without gating mechanism."""
    model = AttentionLSTMModel(
        num_past_features=NUM_PAST_FEATURES,
        num_future_features=NUM_FUTURE_FEATURES,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.0,
        use_gating=False,
    )

    output = model(past_covariates, future_covariates, decoder_lengths)

    assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)
    assert torch.isfinite(output).all()


def test_attention_lstm_different_batch_sizes() -> None:
    """Test AttentionLSTM model with different batch sizes."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.0,
        use_gating=True,
    )

    # Test with different batch sizes
    for batch_size in [1, 4, 8]:
        past = torch.randn(batch_size, 20, 3)
        future = torch.randn(batch_size, 10, 2)
        torch.tensor([20] * batch_size)
        decoder_lengths = torch.tensor([10] * batch_size)

        output = model(past, future, decoder_lengths)

        assert output.shape == (batch_size, 10, 1)
        assert torch.isfinite(output).all()


def test_attention_lstm_deterministic() -> None:
    """Test that AttentionLSTM model is deterministic in eval mode."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.0,  # No dropout for deterministic behavior
        use_gating=False,
    )
    model.eval()

    past = torch.randn(2, 15, 3)
    future = torch.randn(2, 8, 2)
    torch.tensor([15, 15])
    decoder_lengths = torch.tensor([8, 8])

    # Run twice
    with torch.no_grad():
        output1 = model(past, future, decoder_lengths)
        output2 = model(past, future, decoder_lengths)

    # Should be identical
    assert torch.allclose(output1, output2, atol=1e-6)


def test_attention_lstm_gradient_flow() -> None:
    """Test that gradients flow through AttentionLSTM model."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    past = torch.randn(2, 15, 3)
    future = torch.randn(2, 8, 2)
    decoder_lengths = torch.tensor([8, 8])

    output = model(past, future, decoder_lengths)
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed for model parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient for parameter {name}"
        )


def test_attention_lstm_variable_sequence_lengths() -> None:
    """Test AttentionLSTM with variable sequence lengths."""
    model = AttentionLSTMModel(
        num_past_features=2,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.0,
        use_gating=True,
    )

    batch_size = 3
    max_past_len = 20
    max_future_len = 10

    past = torch.randn(batch_size, max_past_len, 2)
    future = torch.randn(batch_size, max_future_len, 2)

    # Variable lengths (only decoder_lengths are used by AttentionLSTM)
    decoder_lengths = torch.tensor([8, 10, 7])

    output = model(past, future, decoder_lengths)

    assert output.shape == (batch_size, max_future_len, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "edge_case",
    [
        "single_element",
        "large_batch",
        "minimal_heads",
    ],
)
def test_attention_lstm_edge_cases(edge_case: str) -> None:
    """Test AttentionLSTM model behavior with edge cases."""
    if edge_case == "single_element":
        model = AttentionLSTMModel(
            num_past_features=2,
            num_future_features=2,
            hidden_size=16,
            num_layers=1,
            n_heads=1,
            dropout=0.0,
            use_gating=False,
        )
        past = torch.randn(1, 5, 2)
        future = torch.randn(1, 3, 2)
        torch.tensor([5])
        decoder_lengths = torch.tensor([3])

        output = model(past, future, decoder_lengths)
        assert torch.isfinite(output).all()

    elif edge_case == "large_batch":
        model = AttentionLSTMModel(
            num_past_features=2,
            num_future_features=2,
            hidden_size=16,
            num_layers=1,
            n_heads=2,
            dropout=0.0,
            use_gating=True,
        )
        batch_size = 64
        past = torch.randn(batch_size, 10, 2)
        future = torch.randn(batch_size, 5, 2)
        torch.tensor([10] * batch_size)
        decoder_lengths = torch.tensor([5] * batch_size)

        output = model(past, future, decoder_lengths)
        assert torch.isfinite(output).all()

    elif edge_case == "minimal_heads":
        model = AttentionLSTMModel(
            num_past_features=2,
            num_future_features=2,
            hidden_size=16,
            num_layers=1,
            n_heads=1,
            dropout=0.0,
            use_gating=True,
        )
        past = torch.randn(4, 10, 2)
        future = torch.randn(4, 5, 2)
        torch.tensor([10] * 4)
        decoder_lengths = torch.tensor([5] * 4)

        output = model(past, future, decoder_lengths)
        assert torch.isfinite(output).all()
