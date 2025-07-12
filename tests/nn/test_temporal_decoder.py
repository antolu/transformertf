from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from transformertf.nn import TemporalDecoder


@pytest.fixture(scope="module")
def sample_compressed_input() -> torch.Tensor:
    """Sample compressed input tensor for testing."""
    return torch.randn(4, 25, 64)  # [batch, compressed_seq_len, hidden_dim]


class TestTemporalDecoder:
    """Test suite for TemporalDecoder."""

    def test_basic_creation(self):
        """Test basic TemporalDecoder creation."""
        decoder = TemporalDecoder(
            input_dim=64,
            output_dim=1,
            target_length=100,
            hidden_dim=32,
            expansion_factor=4,
        )
        assert decoder is not None
        assert decoder.input_dim == 64
        assert decoder.output_dim == 1
        assert decoder.target_length == 100
        assert decoder.hidden_dim == 32
        assert decoder.expansion_factor == 4

    def test_forward_pass_basic(self, sample_compressed_input):
        """Test basic forward pass with expansion."""
        target_length = 100
        decoder = TemporalDecoder(
            input_dim=64,
            output_dim=1,
            target_length=target_length,
            hidden_dim=32,
            expansion_factor=4,
        )

        output = decoder(sample_compressed_input)

        # Should expand to target length
        assert output.shape == (4, target_length, 1)
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()

    def test_expansion_factors(self):
        """Test different expansion factors."""
        compressed_input = torch.randn(2, 50, 32)

        for expansion_factor in [1, 2, 4, 8]:
            target_length = 200
            decoder = TemporalDecoder(
                input_dim=32,
                output_dim=1,
                target_length=target_length,
                hidden_dim=16,
                expansion_factor=expansion_factor,
            )

            output = decoder(compressed_input)

            assert output.shape == (2, target_length, 1)
            assert torch.isfinite(output).all()

    def test_no_expansion(self):
        """Test decoder with expansion_factor=1 (no expansion)."""
        compressed_input = torch.randn(2, 50, 32)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=2,
            target_length=50,  # Same as input length
            hidden_dim=16,
            expansion_factor=1,
        )

        output = decoder(compressed_input)

        assert output.shape == (2, 50, 2)
        assert torch.isfinite(output).all()

    def test_different_target_lengths(self):
        """Test different target lengths."""
        compressed_input = torch.randn(2, 25, 32)

        for target_length in [50, 100, 200, 150]:  # Various target lengths
            decoder = TemporalDecoder(
                input_dim=32,
                output_dim=1,
                target_length=target_length,
                hidden_dim=16,
                expansion_factor=4,
            )

            output = decoder(compressed_input)

            assert output.shape == (2, target_length, 1)
            assert torch.isfinite(output).all()

    def test_multiple_output_dimensions(self):
        """Test decoder with multiple output dimensions."""
        compressed_input = torch.randn(2, 30, 32)

        for output_dim in [1, 3, 7, 10]:
            decoder = TemporalDecoder(
                input_dim=32,
                output_dim=output_dim,
                target_length=120,
                hidden_dim=16,
                expansion_factor=4,
            )

            output = decoder(compressed_input)

            assert output.shape == (2, 120, output_dim)
            assert torch.isfinite(output).all()

    def test_exact_length_interpolation(self):
        """Test that interpolation gives exact target length."""
        compressed_input = torch.randn(2, 33, 32)  # Odd length
        target_length = 127  # Prime number target length

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=target_length,
            hidden_dim=16,
            expansion_factor=4,
        )

        output = decoder(compressed_input)

        # Should be exactly the target length despite odd input/target sizes
        assert output.shape == (2, target_length, 1)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the decoder."""
        compressed_input = torch.randn(2, 25, 32, requires_grad=True)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=100,
            hidden_dim=16,
            expansion_factor=4,
        )

        output = decoder(compressed_input)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert compressed_input.grad is not None
        assert torch.isfinite(compressed_input.grad).all()

        # Check parameter gradients
        for name, param in decoder.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_deterministic_behavior(self):
        """Test deterministic behavior in eval mode."""
        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=100,
            hidden_dim=16,
            expansion_factor=4,
            dropout=0.0,
        )
        decoder.eval()

        compressed_input = torch.randn(2, 25, 32)

        with torch.no_grad():
            output1 = decoder(compressed_input)
            output2 = decoder(compressed_input)

        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
    def test_different_activations(self, activation):
        """Test different activation functions."""
        compressed_input = torch.randn(2, 25, 32)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=100,
            hidden_dim=16,
            activation=activation,
            expansion_factor=4,
        )

        output = decoder(compressed_input)

        assert output.shape == (2, 100, 1)
        assert torch.isfinite(output).all()

    def test_different_num_layers(self):
        """Test different numbers of convolution layers."""
        compressed_input = torch.randn(2, 25, 32)

        for num_layers in [2, 4, 6]:
            decoder = TemporalDecoder(
                input_dim=32,
                output_dim=1,
                target_length=100,
                hidden_dim=16,
                num_layers=num_layers,
                expansion_factor=4,
            )

            output = decoder(compressed_input)

            assert output.shape == (2, 100, 1)
            assert torch.isfinite(output).all()

    def test_batch_independence(self):
        """Test that batch elements are processed independently."""
        # Create input where each batch element is different
        compressed_input = torch.zeros(3, 20, 16)
        compressed_input[0] = 1.0
        compressed_input[1] = -1.0
        compressed_input[2] = 0.5

        decoder = TemporalDecoder(
            input_dim=16,
            output_dim=1,
            target_length=80,
            hidden_dim=8,
            expansion_factor=4,
            dropout=0.0,
        )
        decoder.eval()

        with torch.no_grad():
            output = decoder(compressed_input)

        # Outputs for different batch elements should be different
        assert not torch.allclose(output[0], output[1], atol=1e-3)
        assert not torch.allclose(output[1], output[2], atol=1e-3)
        assert torch.isfinite(output).all()

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        compressed_len=st.integers(min_value=10, max_value=50),
        input_dim=st.integers(min_value=8, max_value=64),
        output_dim=st.integers(min_value=1, max_value=5),
        target_length=st.integers(min_value=40, max_value=200),
        expansion_factor=st.sampled_from([2, 4]),
    )
    @settings(max_examples=15, deadline=None)
    def test_temporal_decoder_properties(
        self,
        batch_size,
        compressed_len,
        input_dim,
        output_dim,
        target_length,
        expansion_factor,
    ):
        """Property-based test for TemporalDecoder invariants."""
        decoder = TemporalDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            target_length=target_length,
            hidden_dim=16,
            expansion_factor=expansion_factor,
            dropout=0.0,
        )

        compressed_input = torch.randn(batch_size, compressed_len, input_dim)
        output = decoder(compressed_input)

        # Property: output should have exact target shape
        assert output.shape == (batch_size, target_length, output_dim)

        # Property: output should be finite
        assert torch.isfinite(output).all()

        # Property: expansion should work as expected
        assert output.shape[1] == target_length

    def test_edge_case_very_short_compressed(self):
        """Test behavior with very short compressed input."""
        # Very short compressed sequence
        compressed_input = torch.randn(2, 3, 32)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=50,
            hidden_dim=16,
            expansion_factor=4,
        )

        output = decoder(compressed_input)

        assert output.shape == (2, 50, 1)
        assert torch.isfinite(output).all()

    def test_edge_case_single_timestep_compressed(self):
        """Test behavior with single timestep compressed input."""
        # Single timestep compressed sequence
        compressed_input = torch.randn(2, 1, 32)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=20,
            hidden_dim=16,
            expansion_factor=4,
        )

        output = decoder(compressed_input)

        assert output.shape == (2, 20, 1)
        assert torch.isfinite(output).all()

    def test_device_consistency(self):
        """Test that output device matches input device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            compressed_input = torch.randn(2, 25, 32, device=device)

            decoder = TemporalDecoder(
                input_dim=32,
                output_dim=1,
                target_length=100,
                hidden_dim=16,
                expansion_factor=4,
            ).to(device)

            output = decoder(compressed_input)

            assert output.device == device
            assert torch.isfinite(output).all()
        else:
            pytest.skip("CUDA not available")

    def test_large_expansion_ratio(self):
        """Test with very large expansion ratio."""
        compressed_input = torch.randn(2, 10, 32)  # Very compressed
        target_length = 800  # Very long target

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=target_length,
            hidden_dim=16,
            expansion_factor=16,  # Large expansion
        )

        output = decoder(compressed_input)

        assert output.shape == (2, target_length, 1)
        assert torch.isfinite(output).all()

    def test_parameter_count_scaling(self):
        """Test that parameter count scales reasonably with dimensions."""
        decoder_small = TemporalDecoder(
            input_dim=16,
            output_dim=1,
            target_length=50,
            hidden_dim=8,
            num_layers=2,
        )

        decoder_large = TemporalDecoder(
            input_dim=64,
            output_dim=3,
            target_length=200,
            hidden_dim=32,
            num_layers=4,
        )

        params_small = sum(p.numel() for p in decoder_small.parameters())
        params_large = sum(p.numel() for p in decoder_large.parameters())

        # Larger decoder should have more parameters
        assert params_large > params_small
        assert params_small > 0
        assert params_large > 0

    def test_output_normalization(self):
        """Test that output normalization is applied."""
        compressed_input = torch.randn(2, 25, 32)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=100,
            hidden_dim=16,
            expansion_factor=4,
        )

        output = decoder(compressed_input)

        # Output should be finite and not have extreme values
        # (due to layer normalization)
        assert torch.isfinite(output).all()
        assert output.abs().max() < 100  # Reasonable bounds

    def test_upsampling_vs_interpolation_consistency(self):
        """Test that upsampling and interpolation work together."""
        compressed_input = torch.randn(2, 23, 32)  # Odd length
        target_length = 97  # Prime number target

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=target_length,
            hidden_dim=16,
            expansion_factor=4,
        )

        output = decoder(compressed_input)

        # Should handle odd input/target lengths gracefully
        assert output.shape == (2, target_length, 1)
        assert torch.isfinite(output).all()

    def test_reconstruction_quality(self):
        """Test that decoder can reconstruct reasonable outputs."""
        # Create structured compressed input
        compressed_input = torch.zeros(1, 10, 16)
        compressed_input[0, :, 0] = torch.linspace(-1, 1, 10)  # Linear trend

        decoder = TemporalDecoder(
            input_dim=16,
            output_dim=1,
            target_length=40,
            hidden_dim=8,
            expansion_factor=4,
            dropout=0.0,
        )
        decoder.eval()

        with torch.no_grad():
            output = decoder(compressed_input)

        assert output.shape == (1, 40, 1)
        assert torch.isfinite(output).all()
        # Output should vary (not constant) given structured input
        assert output.std() > 1e-6

    def test_memory_efficiency_long_target(self):
        """Test memory efficiency with very long target sequences."""
        compressed_input = torch.randn(2, 50, 32)

        decoder = TemporalDecoder(
            input_dim=32,
            output_dim=1,
            target_length=2000,  # Very long target
            hidden_dim=16,
            expansion_factor=8,
        )

        output = decoder(compressed_input)

        assert output.shape == (2, 2000, 1)
        assert torch.isfinite(output).all()

    def test_different_kernel_sizes(self):
        """Test decoder with different kernel sizes in conv blocks."""
        compressed_input = torch.randn(2, 25, 32)

        for kernel_size in [3, 5, 7]:
            decoder = TemporalDecoder(
                input_dim=32,
                output_dim=1,
                target_length=100,
                hidden_dim=16,
                kernel_size=kernel_size,
                expansion_factor=4,
            )

            output = decoder(compressed_input)

            assert output.shape == (2, 100, 1)
            assert torch.isfinite(output).all()
