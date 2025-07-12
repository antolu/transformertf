from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from transformertf.models.temporal_conv_transformer import (
    TCT,
    TemporalConvTransformer,
    TemporalConvTransformerModel,
)


@pytest.fixture(scope="module")
def sample_batch():
    """Sample batch with sufficient sequence lengths."""
    return {
        "encoder_input": torch.randn(4, 400, 10),  # Long enough for compression
        "decoder_input": torch.randn(4, 100, 5),
        "encoder_lengths": torch.full((4, 1), 400),
        "decoder_lengths": torch.full((4, 1), 100),
    }


@pytest.fixture(scope="module")
def short_batch():
    """Sample batch with short sequences for testing warnings."""
    return {
        "encoder_input": torch.randn(2, 50, 8),  # Short sequences
        "decoder_input": torch.randn(2, 20, 4),
        "encoder_lengths": torch.full((2, 1), 50),
        "decoder_lengths": torch.full((2, 1), 20),
    }


class TestTemporalConvTransformerForwardPass:
    """Test suite for TCT forward pass functionality."""

    def test_basic_forward_pass_core_model(self, sample_batch):
        """Test basic forward pass of core TemporalConvTransformerModel."""
        model = TemporalConvTransformerModel(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=64,
            compression_factor=4,
        )

        output = model(
            encoder_input=sample_batch["encoder_input"],
            decoder_input=sample_batch["decoder_input"],
            encoder_lengths=sample_batch.get("encoder_lengths"),
            decoder_lengths=sample_batch.get("decoder_lengths"),
        )

        assert isinstance(output, dict)
        assert "output" in output
        assert "attention_weights" in output

        # Check output shape
        expected_shape = (4, 100, 1)  # batch_size, decoder_seq_len, output_dim
        assert output["output"].shape == expected_shape

        # Check values are finite
        assert torch.isfinite(output["output"]).all()
        assert not torch.isnan(output["output"]).any()

    def test_basic_forward_pass_lightning_model(self, sample_batch):
        """Test basic forward pass of Lightning TemporalConvTransformer."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=64,
            compression_factor=4,
        )

        output = model(
            encoder_input=sample_batch["encoder_input"],
            decoder_input=sample_batch["decoder_input"],
            encoder_lengths=sample_batch.get("encoder_lengths"),
            decoder_lengths=sample_batch.get("decoder_lengths"),
        )

        assert isinstance(output, dict)
        assert "output" in output
        assert "attention_weights" in output

        # Check output shape
        expected_shape = (4, 100, 1)
        assert output["output"].shape == expected_shape
        assert torch.isfinite(output["output"]).all()

    def test_tct_alias_forward_pass(self, sample_batch):
        """Test forward pass using TCT alias."""
        model = TCT(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=32,
            compression_factor=2,
        )

        output = model(
            encoder_input=sample_batch["encoder_input"],
            decoder_input=sample_batch["decoder_input"],
            encoder_lengths=sample_batch.get("encoder_lengths"),
            decoder_lengths=sample_batch.get("decoder_lengths"),
        )

        assert isinstance(output, dict)
        assert "output" in output
        assert output["output"].shape == (4, 100, 1)
        assert torch.isfinite(output["output"]).all()

    @pytest.mark.parametrize("compression_factor", [2, 4, 8])
    def test_different_compression_factors(self, compression_factor):
        """Test forward pass with different compression factors."""
        # Use longer sequences for higher compression factors
        seq_len = 400 if compression_factor <= 4 else 800

        batch = {
            "encoder_input": torch.randn(2, seq_len, 8),
            "decoder_input": torch.randn(2, seq_len // 4, 4),
        }

        model = TemporalConvTransformerModel(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=compression_factor,
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        expected_shape = (2, seq_len // 4, 1)
        assert output["output"].shape == expected_shape
        assert torch.isfinite(output["output"]).all()

    @pytest.mark.parametrize("output_dim", [1, 3, 5, 7])
    def test_different_output_dimensions(self, output_dim):
        """Test forward pass with different output dimensions."""
        batch = {
            "encoder_input": torch.randn(2, 200, 6),
            "decoder_input": torch.randn(2, 50, 3),
        }

        model = TemporalConvTransformerModel(
            num_past_features=6,
            num_future_features=3,
            output_dim=output_dim,
            hidden_dim=32,
            compression_factor=2,
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        expected_shape = (2, 50, output_dim)
        assert output["output"].shape == expected_shape
        assert torch.isfinite(output["output"]).all()

    def test_attention_weights_shape(self, sample_batch):
        """Test that attention weights have correct shape."""
        model = TemporalConvTransformerModel(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=64,
            num_attention_heads=8,
            compression_factor=4,
        )

        output = model(
            encoder_input=sample_batch["encoder_input"],
            decoder_input=sample_batch["decoder_input"],
            encoder_lengths=sample_batch.get("encoder_lengths"),
            decoder_lengths=sample_batch.get("decoder_lengths"),
        )

        attention_weights = output["attention_weights"]

        # Attention weights shape: (batch, decoder_compressed_len, num_heads, total_compressed_len)
        batch_size = 4
        num_heads = 8
        compressed_decoder_len = 100 // 4  # 25
        compressed_encoder_len = 400 // 4  # 100
        total_compressed_len = compressed_encoder_len + compressed_decoder_len  # 125

        expected_shape = (
            batch_size,
            compressed_decoder_len,
            num_heads,
            total_compressed_len,
        )
        assert attention_weights.shape == expected_shape
        assert torch.isfinite(attention_weights).all()

    def test_dynamic_decoder_creation(self):
        """Test that decoder is created dynamically for different target lengths."""
        model = TemporalConvTransformerModel(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=2,
        )

        # Test with different decoder lengths
        for decoder_len in [50, 75, 100]:
            batch = {
                "encoder_input": torch.randn(2, 200, 8),
                "decoder_input": torch.randn(2, decoder_len, 4),
            }

            output = model(
                encoder_input=batch["encoder_input"],
                decoder_input=batch["decoder_input"],
                encoder_lengths=batch.get("encoder_lengths"),
                decoder_lengths=batch.get("decoder_lengths"),
            )

            expected_shape = (2, decoder_len, 1)
            assert output["output"].shape == expected_shape
            assert torch.isfinite(output["output"]).all()

    def test_gradient_flow(self, sample_batch):
        """Test that gradients flow through the entire model."""
        model = TemporalConvTransformerModel(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
        )

        # Make inputs require gradients
        for key in ["encoder_input", "decoder_input"]:
            sample_batch[key] = sample_batch[key].requires_grad_(True)

        output = model(
            encoder_input=sample_batch["encoder_input"],
            decoder_input=sample_batch["decoder_input"],
            encoder_lengths=sample_batch.get("encoder_lengths"),
            decoder_lengths=sample_batch.get("decoder_lengths"),
        )
        loss = output["output"].sum()
        loss.backward()

        # Check input gradients
        for key in ["encoder_input", "decoder_input"]:
            assert sample_batch[key].grad is not None
            assert torch.isfinite(sample_batch[key].grad).all()

        # Check parameter gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_deterministic_behavior(self, sample_batch):
        """Test deterministic behavior in eval mode."""
        model = TemporalConvTransformerModel(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            output1 = model(
                encoder_input=sample_batch["encoder_input"],
                decoder_input=sample_batch["decoder_input"],
                encoder_lengths=sample_batch.get("encoder_lengths"),
                decoder_lengths=sample_batch.get("decoder_lengths"),
            )
            output2 = model(
                encoder_input=sample_batch["encoder_input"],
                decoder_input=sample_batch["decoder_input"],
                encoder_lengths=sample_batch.get("encoder_lengths"),
                decoder_lengths=sample_batch.get("decoder_lengths"),
            )

        assert torch.allclose(output1["output"], output2["output"], atol=1e-6)
        assert torch.allclose(
            output1["attention_weights"], output2["attention_weights"], atol=1e-6
        )

    def test_batch_size_independence(self):
        """Test that different batch sizes work correctly."""
        model = TemporalConvTransformerModel(
            num_past_features=6,
            num_future_features=3,
            output_dim=1,
            hidden_dim=32,
            compression_factor=2,
        )

        for batch_size in [1, 4, 8, 16]:
            batch = {
                "encoder_input": torch.randn(batch_size, 200, 6),
                "decoder_input": torch.randn(batch_size, 50, 3),
            }

            output = model(
                encoder_input=batch["encoder_input"],
                decoder_input=batch["decoder_input"],
                encoder_lengths=batch.get("encoder_lengths"),
                decoder_lengths=batch.get("decoder_lengths"),
            )

            expected_shape = (batch_size, 50, 1)
            assert output["output"].shape == expected_shape
            assert torch.isfinite(output["output"]).all()

    def test_sequence_length_warnings(self, short_batch):
        """Test that warnings are issued for short sequences."""
        model = TemporalConvTransformerModel(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = model(
                encoder_input=short_batch["encoder_input"],
                decoder_input=short_batch["decoder_input"],
                encoder_lengths=short_batch.get("encoder_lengths"),
                decoder_lengths=short_batch.get("decoder_lengths"),
            )

            # Should have issued warnings about short sequences
            assert len(w) >= 1
            assert any(issubclass(warning.category, RuntimeWarning) for warning in w)
            assert any(
                "shorter than recommended" in str(warning.message) for warning in w
            )

        # Should still produce output despite warnings
        assert torch.isfinite(output["output"]).all()

    def test_optional_sequence_lengths(self):
        """Test forward pass without sequence length tensors."""
        batch = {
            "encoder_input": torch.randn(2, 200, 8),
            "decoder_input": torch.randn(2, 50, 4),
            # No encoder_lengths or decoder_lengths
        }

        model = TemporalConvTransformerModel(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=2,
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        assert output["output"].shape == (2, 50, 1)
        assert torch.isfinite(output["output"]).all()

    def test_device_consistency(self):
        """Test that output device matches input device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            batch = {
                "encoder_input": torch.randn(2, 200, 8, device=device),
                "decoder_input": torch.randn(2, 50, 4, device=device),
            }

            model = TemporalConvTransformerModel(
                num_past_features=8,
                num_future_features=4,
                output_dim=1,
                hidden_dim=32,
                compression_factor=2,
            ).to(device)

            output = model(
                encoder_input=batch["encoder_input"],
                decoder_input=batch["decoder_input"],
                encoder_lengths=batch.get("encoder_lengths"),
                decoder_lengths=batch.get("decoder_lengths"),
            )

            assert output["output"].device == device
            assert output["attention_weights"].device == device
            assert torch.isfinite(output["output"]).all()
        else:
            pytest.skip("CUDA not available")

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        encoder_len=st.integers(min_value=200, max_value=600),
        decoder_len=st.integers(min_value=50, max_value=150),
        num_past_features=st.integers(min_value=2, max_value=10),
        num_future_features=st.integers(min_value=1, max_value=8),
        compression_factor=st.sampled_from([2, 4]),
    )
    @settings(max_examples=10, deadline=None)
    def test_forward_pass_properties(
        self,
        batch_size,
        encoder_len,
        decoder_len,
        num_past_features,
        num_future_features,
        compression_factor,
    ):
        """Property-based test for forward pass invariants."""
        model = TemporalConvTransformerModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            output_dim=1,
            hidden_dim=32,
            compression_factor=compression_factor,
        )

        batch = {
            "encoder_input": torch.randn(batch_size, encoder_len, num_past_features),
            "decoder_input": torch.randn(batch_size, decoder_len, num_future_features),
        }

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        # Property: output should have correct shape
        assert output["output"].shape == (batch_size, decoder_len, 1)

        # Property: output should be finite
        assert torch.isfinite(output["output"]).all()

        # Property: attention weights should be finite
        assert torch.isfinite(output["attention_weights"]).all()

    def test_edge_case_single_batch(self):
        """Test edge case with single batch element."""
        batch = {
            "encoder_input": torch.randn(1, 200, 5),
            "decoder_input": torch.randn(1, 50, 3),
        }

        model = TemporalConvTransformerModel(
            num_past_features=5,
            num_future_features=3,
            output_dim=1,
            hidden_dim=32,
            compression_factor=2,
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        assert output["output"].shape == (1, 50, 1)
        assert torch.isfinite(output["output"]).all()

    def test_edge_case_very_long_sequences(self):
        """Test edge case with very long sequences."""
        batch = {
            "encoder_input": torch.randn(2, 2000, 8),  # Very long
            "decoder_input": torch.randn(2, 500, 4),  # Very long
        }

        model = TemporalConvTransformerModel(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=8,  # High compression for efficiency
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        assert output["output"].shape == (2, 500, 1)
        assert torch.isfinite(output["output"]).all()

    def test_edge_case_minimal_sequences(self):
        """Test edge case with minimal viable sequences."""
        # Use sequences just above minimum requirements
        batch = {
            "encoder_input": torch.randn(2, 100, 4),  # Minimal for compression_factor=2
            "decoder_input": torch.randn(2, 25, 2),  # Minimal
        }

        model = TemporalConvTransformerModel(
            num_past_features=4,
            num_future_features=2,
            output_dim=1,
            hidden_dim=16,
            compression_factor=2,
            max_dilation=4,  # Smaller dilation for minimal sequences
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        assert output["output"].shape == (2, 25, 1)
        assert torch.isfinite(output["output"]).all()

    def test_output_consistency_across_runs(self):
        """Test that output is consistent across multiple runs."""
        model = TemporalConvTransformerModel(
            num_past_features=6,
            num_future_features=3,
            output_dim=1,
            hidden_dim=32,
            compression_factor=2,
            dropout=0.0,
        )
        model.eval()

        batch = {
            "encoder_input": torch.randn(2, 100, 6),
            "decoder_input": torch.randn(2, 25, 3),
        }

        outputs = []
        with torch.no_grad():
            for _ in range(3):
                output = model(
                    encoder_input=batch["encoder_input"],
                    decoder_input=batch["decoder_input"],
                    encoder_lengths=batch.get("encoder_lengths"),
                    decoder_lengths=batch.get("decoder_lengths"),
                )
                outputs.append(output["output"])

        # All outputs should be identical in eval mode
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_different_sequence_length_combinations(self):
        """Test various encoder/decoder length combinations."""
        model = TemporalConvTransformerModel(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
        )

        test_cases = [
            (400, 100),  # 4:1 ratio
            (800, 200),  # 4:1 ratio
            (600, 150),  # 4:1 ratio
            (400, 50),  # 8:1 ratio
            (800, 100),  # 8:1 ratio
        ]

        for encoder_len, decoder_len in test_cases:
            batch = {
                "encoder_input": torch.randn(2, encoder_len, 8),
                "decoder_input": torch.randn(2, decoder_len, 4),
            }

            output = model(
                encoder_input=batch["encoder_input"],
                decoder_input=batch["decoder_input"],
                encoder_lengths=batch.get("encoder_lengths"),
                decoder_lengths=batch.get("decoder_lengths"),
            )

            assert output["output"].shape == (2, decoder_len, 1)
            assert torch.isfinite(output["output"]).all()

    def test_with_mocked_attention(self):
        """Test forward pass with mocked attention component."""
        with patch(
            "transformertf.models.temporal_conv_transformer._model.InterpretableMultiHeadAttention"
        ) as mock_attention:
            # Configure mock to return expected shapes
            mock_attention.return_value.forward.return_value = (
                torch.randn(2, 25, 32),  # attended_output
                torch.randn(2, 25, 4, 125),  # attention_weights
            )

            model = TemporalConvTransformerModel(
                num_past_features=8,
                num_future_features=4,
                output_dim=1,
                hidden_dim=32,
                compression_factor=4,
            )

            batch = {
                "encoder_input": torch.randn(2, 400, 8),
                "decoder_input": torch.randn(2, 100, 4),
            }

            output = model(
                encoder_input=batch["encoder_input"],
                decoder_input=batch["decoder_input"],
                encoder_lengths=batch.get("encoder_lengths"),
                decoder_lengths=batch.get("decoder_lengths"),
            )

            # Should complete forward pass with mocked attention
            assert output["output"].shape == (2, 100, 1)
            assert "attention_weights" in output

    def test_memory_efficiency(self):
        """Test memory efficiency with moderately large sequences."""
        batch = {
            "encoder_input": torch.randn(4, 1000, 16),
            "decoder_input": torch.randn(4, 250, 8),
        }

        model = TemporalConvTransformerModel(
            num_past_features=16,
            num_future_features=8,
            output_dim=1,
            hidden_dim=64,
            compression_factor=8,  # High compression for efficiency
        )

        output = model(
            encoder_input=batch["encoder_input"],
            decoder_input=batch["decoder_input"],
            encoder_lengths=batch.get("encoder_lengths"),
            decoder_lengths=batch.get("decoder_lengths"),
        )

        assert output["output"].shape == (4, 250, 1)
        assert torch.isfinite(output["output"]).all()
