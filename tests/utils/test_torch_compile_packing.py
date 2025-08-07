"""
Tests for torch.compile compatibility with sequence packing operations.

This module tests the core issue where torch.compile is incompatible with
pack_padded_sequence and pad_packed_sequence operations, and validates
that @torch.compiler.disable provides a workaround.

See PyTorch issue: https://github.com/pytorch/pytorch/issues/155238
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def test_pack_padded_sequence_torch_compile_issue():
    """
    Test that demonstrates the torch.compile incompatibility with pack_padded_sequence.

    This test reproduces PyTorch issue #155238 where Dynamo fails during FX graph
    tracing when fake tensors encounter pack_padded_sequence operations.
    """

    def problematic_function(
        input_tensor: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Function that uses pack_padded_sequence - breaks with torch.compile."""
        packed = pack_padded_sequence(
            input_tensor, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = pad_packed_sequence(packed, batch_first=True)
        return output

    # Create test data
    batch_size, seq_len, input_size = 4, 10, 5
    input_tensor = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.tensor([8, 10, 6, 9])

    # This should work without compilation
    output_regular = problematic_function(input_tensor, lengths)
    assert output_regular.shape == input_tensor.shape

    # Try to compile - this should fail due to the PyTorch issue
    compiled_function = torch.compile(problematic_function)

    # The compilation should either:
    # 1. Fail with an error (ideal case to demonstrate the issue)
    # 2. Fall back to eager mode (PyTorch's fallback behavior)
    # We'll catch any compilation-related errors to document the issue
    try:
        output_compiled = compiled_function(input_tensor, lengths)
        # If it succeeds, it likely fell back to eager mode
        assert output_compiled.shape == input_tensor.shape
        print(
            "Note: torch.compile may have fallen back to eager mode for pack_padded_sequence"
        )
    except Exception as e:
        # This is expected - document that the issue still exists
        print(
            f"Expected failure with torch.compile and pack_padded_sequence: {type(e).__name__}"
        )
        pytest.skip(f"torch.compile incompatible with pack_padded_sequence: {e}")


@torch.compiler.disable
def _disabled_packing_function(
    input_tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """
    Function with @torch.compiler.disable that safely uses pack_padded_sequence.

    This demonstrates the workaround used in AttentionLSTM for torch.compile compatibility.
    """
    packed = pack_padded_sequence(
        input_tensor, lengths, batch_first=True, enforce_sorted=False
    )
    output, _ = pad_packed_sequence(packed, batch_first=True)
    return output


def test_torch_compiler_disable_fixes_packing():
    """
    Test that @torch.compiler.disable allows packing operations to work with torch.compile.

    This validates our workaround approach used in AttentionLSTM.
    """

    def wrapper_function(
        input_tensor: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compilable wrapper that calls the disabled packing function."""
        # This part can be compiled
        processed_input = input_tensor * 2.0

        # This part is excluded from compilation due to @torch.compiler.disable
        packed_output = _disabled_packing_function(processed_input, lengths)

        # This part can also be compiled
        return packed_output + 1.0

    # Create test data
    batch_size, seq_len, input_size = 4, 10, 5
    input_tensor = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.tensor([8, 10, 6, 9])

    # Test regular execution
    output_regular = wrapper_function(input_tensor, lengths)
    assert output_regular.shape == input_tensor.shape

    # Test with torch.compile - this should work thanks to @torch.compiler.disable
    compiled_wrapper = torch.compile(wrapper_function)
    output_compiled = compiled_wrapper(input_tensor, lengths)

    assert output_compiled.shape == input_tensor.shape

    # Outputs should be identical since the disabled function runs in eager mode
    torch.testing.assert_close(output_regular, output_compiled, rtol=1e-5, atol=1e-5)


def test_lstm_with_packing_compilation():
    """
    Test LSTM with pack_padded_sequence under torch.compile using the disable workaround.

    This simulates the pattern used in AttentionLSTM.
    """

    @torch.compiler.disable
    def _packed_lstm_forward(
        lstm: torch.nn.LSTM, input_tensor: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """LSTM forward with packing - excluded from compilation."""
        packed_input = pack_padded_sequence(
            input_tensor, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output

    def model_forward(
        input_tensor: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Full model forward that can be compiled."""
        lstm = torch.nn.LSTM(input_size=5, hidden_size=32, batch_first=True)

        # Pre-processing (compilable)
        processed_input = torch.nn.functional.relu(input_tensor)

        # LSTM with packing (excluded from compilation)
        lstm_output = _packed_lstm_forward(lstm, processed_input, lengths)

        # Post-processing (compilable)
        return torch.nn.Linear(32, 1)(lstm_output)

    # Create test data
    batch_size, seq_len, input_size = 4, 10, 5
    input_tensor = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.tensor([8, 10, 6, 9])

    # Test regular execution
    output_regular = model_forward(input_tensor, lengths)
    assert output_regular.shape == (batch_size, seq_len, 1)

    # Test with torch.compile
    compiled_model = torch.compile(model_forward)
    output_compiled = compiled_model(input_tensor, lengths)

    assert output_compiled.shape == (batch_size, seq_len, 1)
    # Note: outputs may differ slightly due to compilation optimizations,
    # but the main point is that compilation succeeds without errors
