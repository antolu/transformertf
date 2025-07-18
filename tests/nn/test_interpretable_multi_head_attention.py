from __future__ import annotations

import pytest
import torch

from transformertf.nn import InterpretableMultiHeadAttention


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(2, 200, 4)


def test_interpretable_multi_head_attention(sample: torch.Tensor) -> None:
    model = InterpretableMultiHeadAttention(
        n_dim_model=4,
        n_heads=2,
        dropout=0.1,
    )

    output, _attn = model(sample, sample, sample, return_attn=True)

    assert output.shape == sample.shape


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_features"),
    [
        (1, 10, 2),
        (4, 50, 3),
        (8, 100, 5),
        (16, 200, 2),
    ],
)
def test_attention_output_shapes(batch_size, seq_len, num_features):
    """Test attention module with various input shapes."""
    n_heads = 2 if num_features % 2 == 0 else 1
    attention = InterpretableMultiHeadAttention(
        n_dim_model=num_features,
        n_heads=n_heads,
        dropout=0.1,
    )

    input_tensor = torch.randn(batch_size, seq_len, num_features)
    output, attn_weights = attention(
        input_tensor, input_tensor, input_tensor, return_attn=True
    )

    # Check output shape
    assert output.shape == (batch_size, seq_len, num_features)

    # Check attention weights shape - depends on number of heads
    if n_heads == 1:
        expected_attn_shape = (batch_size, seq_len, seq_len)
    else:
        expected_attn_shape = (batch_size, seq_len, n_heads, seq_len)

    assert attn_weights.shape == expected_attn_shape

    # Check that output is finite and not NaN
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_attention_with_mask():
    """Test attention with causal mask."""
    seq_len = 10
    batch_size = 2
    num_features = 4

    attention = InterpretableMultiHeadAttention(
        n_dim_model=num_features,
        n_heads=2,
        dropout=0.1,
    )

    input_tensor = torch.randn(batch_size, seq_len, num_features)

    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    output = attention(input_tensor, input_tensor, input_tensor, mask=mask)

    assert output.shape == (batch_size, seq_len, num_features)
    assert torch.isfinite(output).all()


def test_attention_different_qkv_shapes():
    """Test attention with different query, key, value shapes."""
    batch_size = 2
    q_len = 20
    kv_len = 30
    num_features = 8

    attention = InterpretableMultiHeadAttention(
        n_dim_model=num_features,
        n_heads=4,
        dropout=0.1,
    )

    query = torch.randn(batch_size, q_len, num_features)
    key = torch.randn(batch_size, kv_len, num_features)
    value = torch.randn(batch_size, kv_len, num_features)

    output, attn_weights = attention(query, key, value, return_attn=True)

    assert output.shape == (batch_size, q_len, num_features)
    assert attn_weights.shape == (batch_size, q_len, 4, kv_len)


def test_attention_initialization():
    """Test that attention weights are properly initialized."""
    attention = InterpretableMultiHeadAttention(
        n_dim_model=16,
        n_heads=4,
        dropout=0.0,
    )

    # Initialize parameters explicitly
    attention.initialize_parameters()

    # Check that parameters are initialized
    for name, param in attention.named_parameters():
        assert param is not None
        assert torch.isfinite(param).all()

        # Bias should be zero after initialization
        if "bias" in name:
            assert torch.allclose(param, torch.zeros_like(param))


def test_attention_deterministic():
    """Test that attention is deterministic when dropout is 0."""
    attention = InterpretableMultiHeadAttention(
        n_dim_model=8,
        n_heads=2,
        dropout=0.0,
    )
    attention.eval()

    input_tensor = torch.randn(2, 10, 8)

    # Run twice with same input
    output1 = attention(input_tensor, input_tensor, input_tensor)
    output2 = attention(input_tensor, input_tensor, input_tensor)

    assert torch.allclose(output1, output2, atol=1e-6)


def test_attention_basic_properties():
    """Test basic attention mechanism properties."""
    attention = InterpretableMultiHeadAttention(
        n_dim_model=64,
        n_heads=8,
        dropout=0.0,
    )

    # Create input tensor with correct feature dimension
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, 64)

    # Basic properties test
    output1, attn_weights = attention(x, x, x, return_attn=True)

    # Check that outputs are finite
    assert torch.isfinite(output1).all()
    assert torch.isfinite(attn_weights).all()

    # Check output shapes
    assert output1.shape == x.shape
    assert attn_weights.shape == (batch_size, seq_len, 8, seq_len)

    # Property: attention should be deterministic when dropout=0
    output2, _ = attention(x, x, x, return_attn=True)
    assert torch.allclose(output1, output2, atol=1e-6)


def test_attention_gradient_flow():
    """Test that gradients flow properly through attention."""
    attention = InterpretableMultiHeadAttention(
        n_dim_model=16,
        n_heads=4,
        dropout=0.1,
    )

    input_tensor = torch.randn(2, 10, 16, requires_grad=True)
    output = attention(input_tensor, input_tensor, input_tensor)

    loss = output.sum()
    loss.backward()

    # Check that gradients are computed
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()

    # Check that attention parameters have gradients
    for name, param in attention.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient for parameter {name}"
        )
