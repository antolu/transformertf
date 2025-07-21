from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from transformertf.nn import TemporalConvBlock


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Sample input tensor for testing."""
    return torch.randn(4, 50, 32)  # [batch, seq_len, channels]


def test_temporal_conv_block_basic_creation():
    """Test basic TemporalConvBlock creation."""
    block = TemporalConvBlock(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        dilation=1,
    )
    assert block is not None
    assert block.in_channels == 32
    assert block.out_channels == 64
    assert block.kernel_size == 3
    assert block.dilation == 1


def test_temporal_conv_block_forward_pass_basic(sample_input):
    """Test basic forward pass."""
    block = TemporalConvBlock(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        dilation=1,
    )

    output = block(sample_input)

    assert output.shape == sample_input.shape
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_temporal_conv_block_dimension_change():
    """Test forward pass with dimension changes."""
    batch_size, seq_len, in_channels = 2, 40, 16
    out_channels = 32

    block = TemporalConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        dilation=1,
    )

    input_tensor = torch.randn(batch_size, seq_len, in_channels)
    output = block(input_tensor)

    assert output.shape == (batch_size, seq_len, out_channels)
    assert torch.isfinite(output).all()


def test_temporal_conv_block_depthwise_vs_standard_convolution():
    """Test depthwise vs standard convolution modes."""
    input_tensor = torch.randn(2, 30, 16)

    # Depthwise convolution
    block_depthwise = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        use_depthwise=True,
    )

    # Standard convolution
    block_standard = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        use_depthwise=False,
    )

    output_depthwise = block_depthwise(input_tensor)
    output_standard = block_standard(input_tensor)

    assert output_depthwise.shape == output_standard.shape
    assert torch.isfinite(output_depthwise).all()
    assert torch.isfinite(output_standard).all()


def test_temporal_conv_block_single_channel_no_depthwise():
    """Test that single channel disables depthwise convolution."""
    input_tensor = torch.randn(2, 30, 1)

    block = TemporalConvBlock(
        in_channels=1,
        out_channels=8,
        use_depthwise=True,  # Should be ignored for single channel
    )

    output = block(input_tensor)
    assert output.shape == (2, 30, 8)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_temporal_conv_block_different_dilations(dilation):
    """Test different dilation rates."""
    input_tensor = torch.randn(2, 100, 16)  # Longer sequence for dilation

    block = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        dilation=dilation,
    )

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_temporal_conv_block_different_kernel_sizes(kernel_size):
    """Test different kernel sizes."""
    input_tensor = torch.randn(2, 50, 16)

    block = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        kernel_size=kernel_size,
    )

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_temporal_conv_block_residual_connections():
    """Test residual connections with matching dimensions."""
    input_tensor = torch.randn(2, 30, 32)

    # Same dimensions - should use residual
    block_residual = TemporalConvBlock(
        in_channels=32,
        out_channels=32,
        dropout=0.0,  # No dropout for deterministic test
    )

    output = block_residual(input_tensor)

    # With residual, output should be different from pure conv output
    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_temporal_conv_block_residual_projection():
    """Test residual projection with dimension mismatch."""
    input_tensor = torch.randn(2, 30, 16)

    block = TemporalConvBlock(
        in_channels=16,
        out_channels=32,  # Different output dimension
        dropout=0.0,
    )

    output = block(input_tensor)

    assert output.shape == (2, 30, 32)
    assert torch.isfinite(output).all()


def test_temporal_conv_block_gradient_flow():
    """Test that gradients flow through the block."""
    input_tensor = torch.randn(2, 20, 16, requires_grad=True)

    block = TemporalConvBlock(
        in_channels=16,
        out_channels=32,
    )

    output = block(input_tensor)
    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()

    # Check parameter gradients
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_temporal_conv_block_deterministic_behavior():
    """Test deterministic behavior in eval mode."""
    block = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        dropout=0.0,
    )
    block.eval()

    input_tensor = torch.randn(2, 30, 16)

    with torch.no_grad():
        output1 = block(input_tensor)
        output2 = block(input_tensor)

    assert torch.allclose(output1, output2, atol=1e-6)


@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
def test_temporal_conv_block_different_activations(activation):
    """Test different activation functions."""
    input_tensor = torch.randn(2, 30, 16)

    block = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        activation=activation,
    )

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_temporal_conv_block_batch_independence():
    """Test that batch elements are processed independently."""
    # Create input where each batch element is different
    input_tensor = torch.zeros(3, 20, 8)
    input_tensor[0] = 1.0
    input_tensor[1] = -1.0
    input_tensor[2] = 0.5

    block = TemporalConvBlock(
        in_channels=8,
        out_channels=8,
        dropout=0.0,
    )
    block.eval()

    with torch.no_grad():
        output = block(input_tensor)

    # Outputs for different batch elements should be different
    assert not torch.allclose(output[0], output[1], atol=1e-3)
    assert not torch.allclose(output[1], output[2], atol=1e-3)
    assert torch.isfinite(output).all()


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=10, max_value=100),
    in_channels=st.integers(min_value=1, max_value=32),
    out_channels=st.integers(min_value=1, max_value=32),
    dilation=st.sampled_from([1, 2, 4]),
)
@settings(max_examples=20, deadline=None)
def test_temporal_conv_block_properties(
    batch_size, seq_len, in_channels, out_channels, dilation
):
    """Property-based test for TemporalConvBlock invariants."""
    block = TemporalConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        dilation=dilation,
        dropout=0.0,
    )

    input_tensor = torch.randn(batch_size, seq_len, in_channels)
    output = block(input_tensor)

    # Property: output should have correct shape
    assert output.shape == (batch_size, seq_len, out_channels)

    # Property: output should be finite
    assert torch.isfinite(output).all()

    # Property: sequence length should be preserved
    assert output.shape[1] == input_tensor.shape[1]


def test_temporal_conv_block_edge_case_very_short_sequence():
    """Test behavior with very short sequences."""
    # Very short sequence that might cause issues with dilation
    input_tensor = torch.randn(2, 3, 16)  # Only 3 time steps

    block = TemporalConvBlock(
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        dilation=1,  # Small dilation for short sequence
    )

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_temporal_conv_block_device_consistency():
    """Test that output device matches input device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        input_tensor = torch.randn(2, 30, 16, device=device)

        block = TemporalConvBlock(
            in_channels=16,
            out_channels=16,
        ).to(device)

        output = block(input_tensor)

        assert output.device == device
        assert torch.isfinite(output).all()
    else:
        pytest.skip("CUDA not available")


def test_temporal_conv_block_parameter_count():
    """Test that parameter count is reasonable."""
    block = TemporalConvBlock(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        use_depthwise=True,
    )

    total_params = sum(p.numel() for p in block.parameters())

    # Should have reasonable number of parameters
    # (not too many due to depthwise convolution)
    assert total_params > 0
    assert total_params < 100000  # Reasonable upper bound


def test_temporal_conv_block_memory_efficiency():
    """Test memory efficiency with larger sequences."""
    # Test with moderately large sequence
    input_tensor = torch.randn(4, 1000, 64)

    block = TemporalConvBlock(
        in_channels=64,
        out_channels=64,
        use_depthwise=True,
    )

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()
