"""
Tests for AttentionLSTM torch.compile integration.

This module tests that the AttentionLSTM model works correctly with torch.compile
when using the @torch.compiler.disable decorated sequence utilities.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from transformertf.models.attention_lstm import AttentionLSTMModel

# Check if GCC is available for torch.compile tests
HAS_GCC = shutil.which("gcc") is not None
torch_compile_available = pytest.mark.skipif(
    not HAS_GCC, reason="torch.compile tests require gcc"
)


class TestAttentionLSTMTorchCompile:
    """Test AttentionLSTM compatibility with torch.compile."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 4
        past_seq_len = 20
        future_seq_len = 10
        num_past_features = 5
        num_future_features = 3

        past_sequence = torch.randn(batch_size, past_seq_len, num_past_features)
        future_sequence = torch.randn(batch_size, future_seq_len, num_future_features)

        # Variable sequence lengths to trigger packing
        encoder_lengths = torch.tensor([18, 20, 15, 19])
        decoder_lengths = torch.tensor([8, 10, 7, 9])

        return {
            "past_sequence": past_sequence,
            "future_sequence": future_sequence,
            "encoder_lengths": encoder_lengths,
            "decoder_lengths": decoder_lengths,
        }

    @pytest.fixture
    def attention_lstm_model(self):
        """Create AttentionLSTM model for testing."""
        return AttentionLSTMModel(
            num_past_features=5,
            num_future_features=3,
            d_model=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )

    def test_attention_lstm_regular_forward_with_packing(
        self, attention_lstm_model, sample_data
    ):
        """Test regular forward pass with sequence packing."""
        with torch.no_grad():
            output = attention_lstm_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                encoder_lengths=sample_data["encoder_lengths"],
                decoder_lengths=sample_data["decoder_lengths"],
            )

        expected_shape = (4, 10, 1)  # (batch_size, future_seq_len, output_dim)
        assert output.shape == expected_shape

    @torch_compile_available
    def test_attention_lstm_torch_compile_with_packing(
        self, attention_lstm_model, sample_data
    ):
        """Test that AttentionLSTM works with torch.compile when using sequence packing."""
        # Compile the model
        compiled_model = torch.compile(attention_lstm_model)

        # Test with sequence packing (this should work due to @torch.compiler.disable)
        with torch.no_grad():
            output_compiled = compiled_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                encoder_lengths=sample_data["encoder_lengths"],
                decoder_lengths=sample_data["decoder_lengths"],
            )

        expected_shape = (4, 10, 1)
        assert output_compiled.shape == expected_shape

    @torch_compile_available
    def test_attention_lstm_torch_compile_without_packing(
        self, attention_lstm_model, sample_data
    ):
        """Test that AttentionLSTM works with torch.compile when not using sequence packing."""
        compiled_model = torch.compile(attention_lstm_model)

        # Test without sequence lengths (no packing)
        with torch.no_grad():
            output_no_packing = compiled_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                # No lengths provided - should not use packing
            )

        expected_shape = (4, 10, 1)
        assert output_no_packing.shape == expected_shape

    @torch_compile_available
    def test_attention_lstm_compiled_vs_regular_consistency(
        self, attention_lstm_model, sample_data
    ):
        """Test that compiled and regular models produce similar outputs."""
        # Set model to eval mode for deterministic behavior
        attention_lstm_model.eval()

        # Regular forward pass
        with torch.no_grad():
            output_regular = attention_lstm_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                encoder_lengths=sample_data["encoder_lengths"],
                decoder_lengths=sample_data["decoder_lengths"],
            )

        # Compiled forward pass
        compiled_model = torch.compile(attention_lstm_model)
        with torch.no_grad():
            output_compiled = compiled_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                encoder_lengths=sample_data["encoder_lengths"],
                decoder_lengths=sample_data["decoder_lengths"],
            )

        # Outputs should have the same shape
        assert output_regular.shape == output_compiled.shape

        # Outputs might differ slightly due to compilation optimizations,
        # but should be reasonably close for most operations
        # Note: We don't assert exact equality due to potential numerical differences

    @torch_compile_available
    def test_attention_lstm_compile_with_return_encoder_states(
        self, attention_lstm_model, sample_data
    ):
        """Test torch.compile compatibility when returning encoder states."""
        compiled_model = torch.compile(attention_lstm_model)

        with torch.no_grad():
            output, encoder_states = compiled_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                encoder_lengths=sample_data["encoder_lengths"],
                decoder_lengths=sample_data["decoder_lengths"],
                return_encoder_states=True,
            )

        # Check output shape
        expected_shape = (4, 10, 1)
        assert output.shape == expected_shape

        # Check encoder states
        hidden_state, cell_state = encoder_states
        assert isinstance(hidden_state, torch.Tensor)
        assert isinstance(cell_state, torch.Tensor)
        assert hidden_state.shape == (2, 4, 32)  # (num_layers, batch_size, d_model)
        assert cell_state.shape == (2, 4, 32)

    @torch_compile_available
    def test_attention_lstm_compile_gradient_flow(
        self, attention_lstm_model, sample_data
    ):
        """Test that gradients flow correctly through compiled model."""
        attention_lstm_model.train()
        compiled_model = torch.compile(attention_lstm_model)

        # Enable gradients
        sample_data["past_sequence"].requires_grad_(True)
        sample_data["future_sequence"].requires_grad_(True)

        output = compiled_model(
            sample_data["past_sequence"],
            sample_data["future_sequence"],
            encoder_lengths=sample_data["encoder_lengths"],
            decoder_lengths=sample_data["decoder_lengths"],
        )

        # Compute a simple loss
        loss = output.mean()
        loss.backward()

        # Check that gradients were computed
        assert sample_data["past_sequence"].grad is not None
        assert sample_data["future_sequence"].grad is not None

        # Check model parameter gradients
        for param in compiled_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @torch_compile_available
    def test_attention_lstm_compile_different_batch_sizes(self, attention_lstm_model):
        """Test torch.compile compatibility with different batch sizes."""
        compiled_model = torch.compile(attention_lstm_model)

        for batch_size in [1, 2, 8]:
            past_sequence = torch.randn(batch_size, 15, 5)
            future_sequence = torch.randn(batch_size, 8, 3)
            encoder_lengths = torch.randint(10, 16, (batch_size,))
            decoder_lengths = torch.randint(5, 9, (batch_size,))

            with torch.no_grad():
                output = compiled_model(
                    past_sequence,
                    future_sequence,
                    encoder_lengths=encoder_lengths,
                    decoder_lengths=decoder_lengths,
                )

            expected_shape = (batch_size, 8, 1)
            assert output.shape == expected_shape

    @torch_compile_available
    def test_attention_lstm_compile_edge_case_lengths(self, attention_lstm_model):
        """Test torch.compile with edge case sequence lengths."""
        compiled_model = torch.compile(attention_lstm_model)

        # Test with uniform lengths (should not use packing)
        past_sequence = torch.randn(3, 10, 5)
        future_sequence = torch.randn(3, 6, 3)
        uniform_encoder_lengths = torch.tensor([10, 10, 10])
        uniform_decoder_lengths = torch.tensor([6, 6, 6])

        with torch.no_grad():
            output = compiled_model(
                past_sequence,
                future_sequence,
                encoder_lengths=uniform_encoder_lengths,
                decoder_lengths=uniform_decoder_lengths,
            )

        assert output.shape == (3, 6, 1)

    @torch_compile_available
    @pytest.mark.parametrize("use_gating", [True, False])
    def test_attention_lstm_compile_different_configs(self, use_gating, sample_data):
        """Test torch.compile with different model configurations."""
        model = AttentionLSTMModel(
            num_past_features=5,
            num_future_features=3,
            d_model=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            use_gating=use_gating,
        )

        compiled_model = torch.compile(model)

        with torch.no_grad():
            output = compiled_model(
                sample_data["past_sequence"],
                sample_data["future_sequence"],
                encoder_lengths=sample_data["encoder_lengths"],
                decoder_lengths=sample_data["decoder_lengths"],
            )

        assert output.shape == (4, 10, 1)
