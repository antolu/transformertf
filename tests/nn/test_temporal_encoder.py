from __future__ import annotations

import warnings

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from transformertf.nn import TemporalEncoder


@pytest.fixture(scope="module")
def sample_input() -> torch.Tensor:
    """Sample input tensor for testing with sufficient length."""
    return torch.randn(4, 200, 32)  # [batch, seq_len, features]


@pytest.fixture(scope="module")
def short_input() -> torch.Tensor:
    """Short input tensor for testing warnings."""
    return torch.randn(2, 30, 16)  # [batch, seq_len, features]


class TestTemporalEncoder:
    """Test suite for TemporalEncoder."""

    def test_basic_creation(self):
        """Test basic TemporalEncoder creation."""
        encoder = TemporalEncoder(
            input_dim=32,
            hidden_dim=64,
            num_layers=4,
            compression_factor=4,
        )
        assert encoder is not None
        assert encoder.input_dim == 32
        assert encoder.hidden_dim == 64
        assert encoder.num_layers == 4
        assert encoder.compression_factor == 4

    def test_forward_pass_basic(self, sample_input):
        """Test basic forward pass with sufficient sequence length."""
        encoder = TemporalEncoder(
            input_dim=32,
            hidden_dim=64,
            compression_factor=4,
        )

        output = encoder(sample_input)

        # Should compress by factor of 4
        expected_length = sample_input.shape[1] // 4
        assert output.shape == (4, expected_length, 64)
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()

    def test_compression_factors(self):
        """Test different compression factors."""
        input_tensor = torch.randn(
            2, 400, 16
        )  # Long enough for all compression factors

        for compression_factor in [1, 2, 4, 8]:
            encoder = TemporalEncoder(
                input_dim=16,
                hidden_dim=32,
                compression_factor=compression_factor,
            )

            output = encoder(input_tensor)

            if compression_factor == 1:
                expected_length = input_tensor.shape[1]
            else:
                expected_length = input_tensor.shape[1] // compression_factor

            assert output.shape[1] == expected_length
            assert output.shape[2] == 32  # hidden_dim
            assert torch.isfinite(output).all()

    def test_no_compression(self):
        """Test encoder with compression_factor=1 (no compression)."""
        input_tensor = torch.randn(2, 100, 16)

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=1,
        )

        output = encoder(input_tensor)

        assert output.shape == (2, 100, 32)
        assert torch.isfinite(output).all()

    def test_sequence_length_warning(self, short_input):
        """Test that warnings are issued for short sequences."""
        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=4,
            max_dilation=8,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = encoder(short_input)

            # Should have issued a RuntimeWarning
            assert len(w) >= 1
            assert any(issubclass(warning.category, RuntimeWarning) for warning in w)
            assert any(
                "shorter than recommended" in str(warning.message) for warning in w
            )

        # Should still produce output despite warning
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[2] == 32  # hidden dim correct
        assert torch.isfinite(output).all()

    def test_different_num_layers(self):
        """Test different numbers of convolution layers."""
        input_tensor = torch.randn(2, 200, 16)

        for num_layers in [2, 4, 6, 8]:
            encoder = TemporalEncoder(
                input_dim=16,
                hidden_dim=32,
                num_layers=num_layers,
                compression_factor=2,
            )

            output = encoder(input_tensor)

            assert output.shape == (2, 100, 32)  # 200 // 2 = 100
            assert torch.isfinite(output).all()

    def test_max_dilation_effects(self):
        """Test different max_dilation values."""
        input_tensor = torch.randn(2, 400, 16)  # Very long sequence

        for max_dilation in [2, 4, 8, 16]:
            encoder = TemporalEncoder(
                input_dim=16,
                hidden_dim=32,
                max_dilation=max_dilation,
                num_layers=4,
                compression_factor=4,
            )

            output = encoder(input_tensor)

            assert output.shape == (2, 100, 32)  # 400 // 4 = 100
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the encoder."""
        input_tensor = torch.randn(2, 100, 16, requires_grad=True)

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=2,
        )

        output = encoder(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()

        # Check parameter gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_deterministic_behavior(self):
        """Test deterministic behavior in eval mode."""
        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=2,
            dropout=0.0,
        )
        encoder.eval()

        input_tensor = torch.randn(2, 100, 16)

        with torch.no_grad():
            output1 = encoder(input_tensor)
            output2 = encoder(input_tensor)

        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
    def test_different_activations(self, activation):
        """Test different activation functions."""
        input_tensor = torch.randn(2, 100, 16)

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            activation=activation,
            compression_factor=2,
        )

        output = encoder(input_tensor)

        assert output.shape == (2, 50, 32)
        assert torch.isfinite(output).all()

    def test_adaptive_pooling_exactness(self):
        """Test that adaptive pooling gives exact compression ratios."""
        # Test with sequence length not evenly divisible by compression factor
        input_tensor = torch.randn(2, 203, 16)  # 203 is not divisible by 4

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=4,
        )

        output = encoder(input_tensor)

        expected_length = 203 // 4  # Should be 50 (floor division)
        assert output.shape == (2, expected_length, 32)
        assert torch.isfinite(output).all()

    def test_batch_independence(self):
        """Test that batch elements are processed independently."""
        # Create input where each batch element is different
        input_tensor = torch.zeros(3, 100, 8)
        input_tensor[0] = 1.0
        input_tensor[1] = -1.0
        input_tensor[2] = 0.5

        encoder = TemporalEncoder(
            input_dim=8,
            hidden_dim=16,
            compression_factor=2,
            dropout=0.0,
        )
        encoder.eval()

        with torch.no_grad():
            output = encoder(input_tensor)

        # Outputs for different batch elements should be different
        assert not torch.allclose(output[0], output[1], atol=1e-3)
        assert not torch.allclose(output[1], output[2], atol=1e-3)
        assert torch.isfinite(output).all()

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=100, max_value=400),  # Long enough sequences
        input_dim=st.integers(min_value=4, max_value=32),
        hidden_dim=st.integers(min_value=8, max_value=64),
        compression_factor=st.sampled_from([2, 4]),
    )
    @settings(max_examples=15, deadline=None)
    def test_temporal_encoder_properties(
        self, batch_size, seq_len, input_dim, hidden_dim, compression_factor
    ):
        """Property-based test for TemporalEncoder invariants."""
        encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            compression_factor=compression_factor,
            dropout=0.0,
        )

        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(input_tensor)

        # Property: output should have correct shape
        expected_seq_len = seq_len // compression_factor
        assert output.shape == (batch_size, expected_seq_len, hidden_dim)

        # Property: output should be finite
        assert torch.isfinite(output).all()

        # Property: compression should work as expected
        assert output.shape[1] <= input_tensor.shape[1]

    def test_edge_case_minimum_sequence(self):
        """Test behavior with minimum viable sequence length."""
        # Test with exactly minimum recommended length
        compression_factor = 2
        max_dilation = 4
        min_length = compression_factor * max_dilation * 3  # Conservative minimum

        input_tensor = torch.randn(2, min_length, 16)

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=compression_factor,
            max_dilation=max_dilation,
        )

        output = encoder(input_tensor)

        expected_length = min_length // compression_factor
        assert output.shape == (2, expected_length, 32)
        assert torch.isfinite(output).all()

    def test_device_consistency(self):
        """Test that output device matches input device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            input_tensor = torch.randn(2, 200, 16, device=device)

            encoder = TemporalEncoder(
                input_dim=16,
                hidden_dim=32,
                compression_factor=4,
            ).to(device)

            output = encoder(input_tensor)

            assert output.device == device
            assert torch.isfinite(output).all()
        else:
            pytest.skip("CUDA not available")

    def test_very_large_compression(self):
        """Test with very large compression factor."""
        input_tensor = torch.randn(2, 800, 16)  # Very long sequence

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=16,  # Very high compression
        )

        output = encoder(input_tensor)

        expected_length = 800 // 16  # Should be 50
        assert output.shape == (2, expected_length, 32)
        assert torch.isfinite(output).all()

    def test_parameter_count_scaling(self):
        """Test that parameter count scales reasonably with dimensions."""
        encoder_small = TemporalEncoder(
            input_dim=8,
            hidden_dim=16,
            num_layers=2,
        )

        encoder_large = TemporalEncoder(
            input_dim=32,
            hidden_dim=64,
            num_layers=4,
        )

        params_small = sum(p.numel() for p in encoder_small.parameters())
        params_large = sum(p.numel() for p in encoder_large.parameters())

        # Larger encoder should have more parameters
        assert params_large > params_small
        assert params_small > 0
        assert params_large > 0

    def test_output_normalization(self):
        """Test that output normalization is applied."""
        input_tensor = torch.randn(2, 100, 16)

        encoder = TemporalEncoder(
            input_dim=16,
            hidden_dim=32,
            compression_factor=2,
        )

        output = encoder(input_tensor)

        # Output should be finite and not have extreme values
        # (due to layer normalization)
        assert torch.isfinite(output).all()
        assert output.abs().max() < 100  # Reasonable bounds

    def test_memory_efficiency_long_sequence(self):
        """Test memory efficiency with very long sequences."""
        # Test with very long sequence
        input_tensor = torch.randn(2, 2000, 32)

        encoder = TemporalEncoder(
            input_dim=32,
            hidden_dim=64,
            compression_factor=8,  # High compression for efficiency
        )

        output = encoder(input_tensor)

        expected_length = 2000 // 8  # Should be 250
        assert output.shape == (2, expected_length, 64)
        assert torch.isfinite(output).all()

    def test_different_kernel_sizes(self):
        """Test encoder with different kernel sizes in conv blocks."""
        input_tensor = torch.randn(2, 200, 16)

        for kernel_size in [3, 5, 7]:
            encoder = TemporalEncoder(
                input_dim=16,
                hidden_dim=32,
                kernel_size=kernel_size,
                compression_factor=4,
            )

            output = encoder(input_tensor)

            assert output.shape == (2, 50, 32)
            assert torch.isfinite(output).all()
