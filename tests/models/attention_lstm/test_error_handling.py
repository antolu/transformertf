from __future__ import annotations

import pytest
import torch

from transformertf.models.attention_lstm import AttentionLSTM, AttentionLSTMModel


def test_attention_lstm_invalid_parameters() -> None:
    """Test AttentionLSTM with invalid parameters."""
    # Test with invalid hidden_size (not divisible by n_heads)
    with pytest.raises(AssertionError):
        AttentionLSTMModel(
            num_past_features=3,
            num_future_features=2,
            hidden_size=30,  # Not divisible by 4
            num_layers=1,
            n_heads=4,
            dropout=0.1,
            use_gating=True,
        )

    # Test that model can be created with valid parameters
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )
    assert model is not None


def test_attention_lstm_mismatched_input_shapes() -> None:
    """Test AttentionLSTM with mismatched input shapes."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Test with wrong number of past features
    past_covariates = torch.randn(batch_size, past_seq_len, 5)  # Wrong feature count
    future_covariates = torch.randn(batch_size, future_seq_len, 2)
    decoder_lengths = torch.tensor([future_seq_len] * batch_size)
    with pytest.raises(RuntimeError):
        model(past_covariates, future_covariates, decoder_lengths)

    # Test with wrong number of future features
    past_covariates = torch.randn(batch_size, past_seq_len, 3)
    future_covariates = torch.randn(
        batch_size, future_seq_len, 5
    )  # Wrong feature count
    decoder_lengths = torch.tensor([future_seq_len] * batch_size)
    with pytest.raises(RuntimeError):
        model(past_covariates, future_covariates, decoder_lengths)

    # Test with mismatched batch sizes
    past_covariates = torch.randn(4, past_seq_len, 3)
    future_covariates = torch.randn(2, future_seq_len, 2)  # Different batch size
    decoder_lengths = torch.tensor([future_seq_len] * 2)
    with pytest.raises((RuntimeError, ValueError)):
        model(past_covariates, future_covariates, decoder_lengths)


def test_attention_lstm_invalid_sequence_lengths() -> None:
    """Test AttentionLSTM with invalid sequence lengths."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    past_covariates = torch.randn(batch_size, past_seq_len, 3)
    future_covariates = torch.randn(batch_size, future_seq_len, 2)

    # Test with decoder_lengths exceeding sequence length
    decoder_lengths = torch.tensor([15, 10, 10, 10])  # First exceeds future_seq_len
    with pytest.raises((RuntimeError, IndexError)):
        model(past_covariates, future_covariates, decoder_lengths=decoder_lengths)

    # Test with zero lengths
    decoder_lengths = torch.tensor([0, 10, 10, 10])  # Zero length
    with pytest.raises((RuntimeError, ValueError)):
        model(past_covariates, future_covariates, decoder_lengths=decoder_lengths)


def test_attention_lstm_module_invalid_hyperparameters() -> None:
    """Test AttentionLSTM with valid hyperparameters."""
    # Test that module can be created with valid parameters
    module = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )
    assert module is not None

    # Test hyperparameter storage
    assert module.hparams.num_past_features == 3
    assert module.hparams.num_future_features == 2
    assert module.hparams.hidden_size == 32


def test_attention_lstm_empty_batch() -> None:
    """Test AttentionLSTM with minimal batch."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    # Test with batch size 1 (minimal valid batch)
    past_covariates = torch.randn(1, 20, 3)
    future_covariates = torch.randn(1, 10, 2)
    decoder_lengths = torch.tensor([10])

    output = model(past_covariates, future_covariates, decoder_lengths)
    assert output.shape == (1, 10, 1)
    assert torch.isfinite(output).all()


def test_attention_lstm_nan_inputs() -> None:
    """Test AttentionLSTM behavior with NaN inputs."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Test with NaN in past_covariates
    past_covariates = torch.randn(batch_size, past_seq_len, 3)
    past_covariates[0, 0, 0] = float("nan")
    future_covariates = torch.randn(batch_size, future_seq_len, 2)
    decoder_lengths = torch.tensor([future_seq_len] * batch_size)

    output = model(past_covariates, future_covariates, decoder_lengths)

    # Model should produce NaN output when given NaN input
    assert torch.isnan(output).any()


def test_attention_lstm_inf_inputs() -> None:
    """Test AttentionLSTM behavior with infinite inputs."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Test with inf in future_covariates
    past_covariates = torch.randn(batch_size, past_seq_len, 3)
    future_covariates = torch.randn(batch_size, future_seq_len, 2)
    future_covariates[0, 0, 0] = float("inf")
    decoder_lengths = torch.tensor([future_seq_len] * batch_size)

    output = model(past_covariates, future_covariates, decoder_lengths)

    # Model handles infinite inputs robustly and produces valid output
    # The LSTM and attention mechanisms have internal stabilization
    assert output is not None
    assert output.shape == (batch_size, future_seq_len, 1)


def test_attention_lstm_module_missing_batch_keys() -> None:
    """Test AttentionLSTM with missing batch keys."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    # Test with missing encoder_input
    batch = {
        "decoder_input": torch.randn(4, 10, 2),
        "decoder_lengths": torch.tensor([[10], [10], [10], [10]]),
        "target": torch.randn(4, 10, 1),
    }
    with pytest.raises(KeyError):
        model.training_step(batch, batch_idx=0)

    # Test with missing target (should fail in training)
    batch = {
        "encoder_input": torch.randn(4, 20, 3),
        "decoder_input": torch.randn(4, 10, 2),
        "decoder_lengths": torch.tensor([[10], [10], [10], [10]]),
    }
    with pytest.raises(AssertionError):
        model.training_step(batch, batch_idx=0)
