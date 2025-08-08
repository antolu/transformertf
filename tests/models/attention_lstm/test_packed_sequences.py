from __future__ import annotations

import torch

from transformertf.models.attention_lstm import AttentionLSTMModel


def test_attention_lstm_basic_functionality():
    """Test that AttentionLSTM model can be created and used."""
    # Create model
    model = AttentionLSTMModel(
        num_past_features=4,
        num_future_features=3,
        d_model=16,
        num_layers=1,
        num_heads=2,
    )

    # Verify model can be created
    assert model is not None
    assert model.d_model == 16
    assert model.num_layers == 1
    assert model.num_heads == 2


def test_basic_forward_pass():
    """Test basic forward pass without variable lengths."""
    model = AttentionLSTMModel(
        num_past_features=4,
        num_future_features=3,
        d_model=32,
        num_layers=2,
        num_heads=4,
        use_gating=False,  # Simpler for testing
    )

    batch_size = 2
    past_len, future_len = 10, 6

    # Create input sequences
    past_sequence = torch.randn(batch_size, past_len, 4)
    future_sequence = torch.randn(batch_size, future_len, 3)

    # Forward pass without length information
    output = model(past_sequence, future_sequence)

    # Verify output shape
    assert output.shape == (batch_size, future_len, model.output_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_variable_length_forward_pass():
    """Test AttentionLSTM with variable sequence lengths."""
    model = AttentionLSTMModel(
        num_past_features=4,
        num_future_features=3,
        d_model=32,
        num_layers=2,
        num_heads=4,
        use_gating=False,  # Simpler for testing
    )

    batch_size = 4
    max_past_len, max_future_len = 12, 8

    # Create input sequences
    past_sequence = torch.randn(batch_size, max_past_len, 4)
    future_sequence = torch.randn(batch_size, max_future_len, 3)

    # Create variable lengths that would benefit from packing
    encoder_lengths = torch.tensor([10, 12, 7, 9])
    decoder_lengths = torch.tensor([6, 8, 5, 7])

    # Forward pass should automatically use packing
    output = model(past_sequence, future_sequence, encoder_lengths, decoder_lengths)

    # Verify output shape
    assert output.shape == (batch_size, max_future_len, model.output_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_uniform_length_forward_pass():
    """Test AttentionLSTM with uniform lengths (should not use packing)."""
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        d_model=24,
        num_layers=1,
        num_heads=3,
    )

    batch_size = 3
    past_len, future_len = 10, 6

    # Create input sequences
    past_sequence = torch.randn(batch_size, past_len, 3)
    future_sequence = torch.randn(batch_size, future_len, 2)

    # All sequences have same length (no packing benefit)
    encoder_lengths = torch.tensor([10, 10, 10])
    decoder_lengths = torch.tensor([6, 6, 6])

    # Forward pass should not use packing
    output = model(past_sequence, future_sequence, encoder_lengths, decoder_lengths)

    # Verify output shape
    assert output.shape == (batch_size, future_len, model.output_dim)


def test_no_lengths_provided():
    """Test AttentionLSTM when no length information is provided."""
    model = AttentionLSTMModel(
        num_past_features=2,
        num_future_features=2,
        d_model=16,
        num_layers=1,
        num_heads=2,
    )

    batch_size = 2
    past_len, future_len = 8, 5

    # Create input sequences
    past_sequence = torch.randn(batch_size, past_len, 2)
    future_sequence = torch.randn(batch_size, future_len, 2)

    # No length information - should fall back to standard processing
    output = model(past_sequence, future_sequence)

    # Verify output shape
    assert output.shape == (batch_size, future_len, model.output_dim)


def test_return_encoder_states():
    """Test that packed sequences work with encoder state returns."""
    model = AttentionLSTMModel(
        num_past_features=2,
        num_future_features=2,
        d_model=16,
        num_layers=1,
        num_heads=2,
    )

    batch_size = 3
    past_len, future_len = 8, 5

    # Create input sequences
    past_sequence = torch.randn(batch_size, past_len, 2)
    future_sequence = torch.randn(batch_size, future_len, 2)

    # Variable lengths
    encoder_lengths = torch.tensor([6, 8, 7])
    decoder_lengths = torch.tensor([3, 5, 4])

    # Test with encoder state return
    output, encoder_states = model(
        past_sequence,
        future_sequence,
        encoder_lengths,
        decoder_lengths,
        return_encoder_states=True,
    )

    # Verify output and states
    assert output.shape == (batch_size, future_len, model.output_dim)
    assert len(encoder_states) == 2  # (h_n, c_n)
    assert encoder_states[0].shape == (model.num_layers, batch_size, model.d_model)
    assert encoder_states[1].shape == (model.num_layers, batch_size, model.d_model)


def test_gradient_flow():
    """Test that gradients flow properly through packed sequences."""
    model = AttentionLSTMModel(
        num_past_features=2,
        num_future_features=2,
        d_model=16,
        num_layers=1,
        num_heads=2,
    )

    batch_size = 2
    past_len, future_len = 6, 4

    # Create input sequences
    past_sequence = torch.randn(batch_size, past_len, 2, requires_grad=True)
    future_sequence = torch.randn(batch_size, future_len, 2, requires_grad=True)

    # Variable lengths to trigger packing
    encoder_lengths = torch.tensor([4, 6])
    decoder_lengths = torch.tensor([3, 4])

    # Forward pass
    output = model(past_sequence, future_sequence, encoder_lengths, decoder_lengths)

    # Create a simple loss
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Verify gradients exist
    assert past_sequence.grad is not None
    assert future_sequence.grad is not None
    assert not torch.isnan(past_sequence.grad).any()
    assert not torch.isnan(future_sequence.grad).any()

    # Verify model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
