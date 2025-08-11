"""
Tests for torch.compile compatibility of sequence utility functions.

This module tests that the @torch.compiler.disable decorated sequence utilities
in transformertf.utils.sequence work correctly with torch.compile.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from transformertf.utils.sequence import (
    align_encoder_sequences,
    pack_decoder_sequences,
    pack_encoder_sequences,
    should_use_packing,
    unpack_to_fixed_length,
    validate_encoder_alignment,
)

# Check if GCC is available for torch.compile tests
HAS_GCC = shutil.which("gcc") is not None
torch_compile_available = pytest.mark.skipif(
    not HAS_GCC, reason="torch.compile tests require gcc"
)


@torch_compile_available
def test_should_use_packing_torch_compile():
    """Test that should_use_packing works with torch.compile."""

    def wrapper_function(lengths: torch.Tensor | None) -> bool:
        return should_use_packing(lengths)

    # Test with variable lengths (should use packing)
    variable_lengths = torch.tensor([10, 8, 12, 6])
    result_regular = wrapper_function(variable_lengths)

    compiled_wrapper = torch.compile(wrapper_function)
    result_compiled = compiled_wrapper(variable_lengths)

    assert result_regular == result_compiled

    # Test with uniform lengths (should not use packing)
    uniform_lengths = torch.tensor([10, 10, 10, 10])
    result_regular_uniform = wrapper_function(uniform_lengths)
    result_compiled_uniform = compiled_wrapper(uniform_lengths)

    assert result_regular_uniform == result_compiled_uniform


@torch_compile_available
def test_pack_encoder_sequences_torch_compile_wrapper():
    """Test pack_encoder_sequences works when called from compiled code."""

    def wrapper_function(sequences: torch.Tensor, lengths: torch.Tensor):
        """Wrapper that calls the @torch.compiler.disable decorated function."""
        packed = pack_encoder_sequences(sequences, lengths, align_first=True)
        # Return properties we can verify
        return packed.data.shape, len(packed.batch_sizes)

    batch_size, seq_len, features = 4, 12, 8
    sequences = torch.randn(batch_size, seq_len, features)
    lengths = torch.tensor([10, 12, 8, 11])

    # Test regular execution
    shape_regular, batch_sizes_len_regular = wrapper_function(sequences, lengths)

    # Test with torch.compile - this should work
    compiled_wrapper = torch.compile(wrapper_function)
    shape_compiled, batch_sizes_len_compiled = compiled_wrapper(sequences, lengths)

    # Results should be identical
    assert shape_regular == shape_compiled
    assert batch_sizes_len_regular == batch_sizes_len_compiled


@torch_compile_available
def test_pack_decoder_sequences_torch_compile_wrapper():
    """Test pack_decoder_sequences works when called from compiled code."""

    def wrapper_function(sequences: torch.Tensor, lengths: torch.Tensor):
        packed = pack_decoder_sequences(sequences, lengths)
        return packed.data.shape, len(packed.batch_sizes)

    batch_size, seq_len, features = 4, 8, 6
    sequences = torch.randn(batch_size, seq_len, features)
    lengths = torch.tensor([6, 8, 5, 7])

    # Test regular execution
    shape_regular, batch_sizes_len_regular = wrapper_function(sequences, lengths)

    # Test with torch.compile
    compiled_wrapper = torch.compile(wrapper_function)
    shape_compiled, batch_sizes_len_compiled = compiled_wrapper(sequences, lengths)

    # Results should be identical
    assert shape_regular == shape_compiled
    assert batch_sizes_len_regular == batch_sizes_len_compiled


@torch_compile_available
def test_unpack_to_fixed_length_torch_compile_wrapper():
    """Test unpack_to_fixed_length works when called from compiled code."""

    def pack_and_unpack_function(sequences: torch.Tensor, lengths: torch.Tensor):
        # First pack the sequences
        packed = pack_encoder_sequences(sequences, lengths)

        # Then unpack them - this calls the @torch.compiler.disable function
        unpacked, unpacked_lengths = unpack_to_fixed_length(
            packed, total_length=sequences.size(1)
        )

        return unpacked.shape, unpacked_lengths.shape

    batch_size, seq_len, features = 4, 12, 8
    sequences = torch.randn(batch_size, seq_len, features)
    lengths = torch.tensor([10, 12, 8, 11])

    # Test regular execution
    unpacked_shape_regular, lengths_shape_regular = pack_and_unpack_function(
        sequences, lengths
    )

    # Test with torch.compile
    compiled_function = torch.compile(pack_and_unpack_function)
    unpacked_shape_compiled, lengths_shape_compiled = compiled_function(
        sequences, lengths
    )

    # Results should be identical
    assert unpacked_shape_regular == unpacked_shape_compiled
    assert lengths_shape_regular == lengths_shape_compiled


@torch_compile_available
def test_full_sequence_pipeline_torch_compile():
    """Test the full sequence processing pipeline with torch.compile."""

    def full_pipeline(sequences: torch.Tensor, lengths: torch.Tensor):
        """Full pipeline: check packing decision, pack, process with LSTM, unpack."""
        use_packing = should_use_packing(lengths)

        if use_packing:
            # Pack sequences
            packed_input = pack_encoder_sequences(sequences, lengths)

            # Process with LSTM
            lstm = torch.nn.LSTM(input_size=sequences.size(-1), hidden_size=16)
            packed_output, _ = lstm(packed_input)

            # Unpack sequences
            output, output_lengths = unpack_to_fixed_length(
                packed_output, total_length=sequences.size(1)
            )

            return output, output_lengths, True
        # Process without packing
        lstm = torch.nn.LSTM(input_size=sequences.size(-1), hidden_size=16)
        output, _ = lstm(sequences)
        return output, lengths, False

    batch_size, seq_len, features = 4, 12, 8
    sequences = torch.randn(batch_size, seq_len, features)
    lengths = torch.tensor([10, 12, 8, 11])

    # Test regular execution
    output_regular, lengths_regular, used_packing_regular = full_pipeline(
        sequences, lengths
    )

    # Test with torch.compile
    compiled_pipeline = torch.compile(full_pipeline)
    output_compiled, lengths_compiled, used_packing_compiled = compiled_pipeline(
        sequences, lengths
    )

    # Results should be consistent
    assert output_regular.shape == output_compiled.shape
    assert lengths_regular.shape == lengths_compiled.shape
    assert used_packing_regular == used_packing_compiled


def test_packing_decision_logic():
    """Test the logic of should_use_packing function."""
    # Test with None - should not use packing
    assert not should_use_packing(None)

    # Test with empty tensor - should not use packing
    assert not should_use_packing(torch.tensor([]))

    # Test with uniform lengths - should not use packing
    uniform_lengths = torch.tensor([10, 10, 10, 10])
    assert not should_use_packing(uniform_lengths)

    # Test with variable lengths - should use packing
    variable_lengths = torch.tensor([10, 8, 12, 6])
    assert should_use_packing(variable_lengths)

    # Test with small variation but large batch - should use packing
    small_variation_large_batch = torch.tensor([10, 11, 10, 11, 10, 11, 10, 11])
    assert should_use_packing(small_variation_large_batch)

    # Test with large variation - should use packing
    large_variation = torch.tensor([20, 10, 15, 5])
    assert should_use_packing(large_variation)


def test_align_encoder_sequences_correctness():
    """Test that align_encoder_sequences correctly moves padding."""
    # Create left-padded sequences (padding at end)
    sequences = torch.tensor(
        [
            [[1, 2], [3, 4], [0, 0]],  # length=2, padding at end
            [[5, 6], [7, 8], [9, 10]],  # length=3, no padding
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2, 3])

    aligned = align_encoder_sequences(sequences, lengths)

    # First sequence should have padding moved to beginning (right-aligned)
    expected_first = torch.tensor([[0, 0], [1, 2], [3, 4]], dtype=torch.float32)
    torch.testing.assert_close(aligned[0], expected_first)

    # Second sequence should remain unchanged (no padding)
    torch.testing.assert_close(aligned[1], sequences[1])


def test_validate_encoder_alignment():
    """Test encoder alignment validation function."""
    # Test left-aligned sequences (padding at end)
    left_aligned = torch.tensor(
        [
            [[1, 2], [3, 4], [0, 0]],  # padding at end
            [[5, 6], [7, 8], [9, 10]],  # no padding
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2, 3])

    # Should not raise for left alignment
    validate_encoder_alignment(left_aligned, lengths, "left")

    # Should raise for right alignment expectation
    with pytest.raises(ValueError, match="Expected right alignment.*padding at start"):
        validate_encoder_alignment(left_aligned, lengths, "right")

    # Test right-aligned sequences (padding at start)
    right_aligned = torch.tensor(
        [
            [[0, 0], [1, 2], [3, 4]],  # padding at start
            [[5, 6], [7, 8], [9, 10]],  # no padding
        ],
        dtype=torch.float32,
    )

    # Should not raise for right alignment
    validate_encoder_alignment(right_aligned, lengths, "right")

    # Should raise for left alignment expectation
    with pytest.raises(ValueError, match="Expected left alignment.*padding at end"):
        validate_encoder_alignment(right_aligned, lengths, "left")


def test_pack_unpack_roundtrip_consistency():
    """Test that pack/unpack operations are consistent."""
    batch_size, seq_len, features = 3, 10, 5
    sequences = torch.randn(batch_size, seq_len, features)
    lengths = torch.tensor([8, 10, 6])

    # Test encoder sequences
    packed_encoder = pack_encoder_sequences(sequences, lengths, align_first=False)
    unpacked_encoder, unpacked_lengths = unpack_to_fixed_length(
        packed_encoder, total_length=seq_len
    )

    assert unpacked_encoder.shape == sequences.shape
    torch.testing.assert_close(unpacked_lengths, lengths)

    # Test decoder sequences
    packed_decoder = pack_decoder_sequences(sequences, lengths)
    unpacked_decoder, unpacked_lengths_dec = unpack_to_fixed_length(
        packed_decoder, total_length=seq_len
    )

    assert unpacked_decoder.shape == sequences.shape
    torch.testing.assert_close(unpacked_lengths_dec, lengths)


def test_pack_sequences_with_edge_cases():
    """Test packing functions with edge case inputs."""
    # Test single sequence
    single_seq = torch.randn(1, 5, 3)
    single_length = torch.tensor([4])

    packed_single = pack_encoder_sequences(single_seq, single_length)
    assert packed_single.data.shape[0] == 4  # Only 4 timesteps should be packed

    # Test all sequences same length
    same_length_seq = torch.randn(3, 8, 4)
    same_lengths = torch.tensor([8, 8, 8])

    packed_same = pack_decoder_sequences(same_length_seq, same_lengths)
    unpacked_same, _ = unpack_to_fixed_length(packed_same, total_length=8)

    # Should be identical since no actual padding was removed
    torch.testing.assert_close(unpacked_same, same_length_seq, rtol=1e-5, atol=1e-6)


def test_sequence_utilities_preserve_gradients():
    """Test that sequence utilities preserve gradient information."""
    batch_size, seq_len, features = 2, 6, 3
    sequences = torch.randn(batch_size, seq_len, features, requires_grad=True)
    lengths = torch.tensor([5, 6])

    # Pack and unpack
    packed = pack_encoder_sequences(sequences, lengths)
    unpacked, _ = unpack_to_fixed_length(packed, total_length=seq_len)

    # Compute a simple loss and backpropagate
    loss = unpacked.sum()
    loss.backward()

    # Original sequences should have gradients
    assert sequences.grad is not None
    assert sequences.grad.shape == sequences.shape


def test_pack_sequences_length_validation():
    """Test that packing functions handle invalid lengths appropriately."""
    sequences = torch.randn(2, 5, 3)

    # Test with lengths longer than sequences
    invalid_lengths = torch.tensor([10, 8])  # Longer than seq_len=5

    # Should still work (pack_padded_sequence handles this)
    packed = pack_encoder_sequences(sequences, invalid_lengths)
    assert packed is not None

    # Test with valid positive lengths only
    valid_lengths = torch.tensor([3, 5])
    packed_valid = pack_decoder_sequences(sequences, valid_lengths)
    assert packed_valid is not None


def test_unpack_different_total_lengths():
    """Test unpacking with different total_length parameters."""
    batch_size, seq_len, features = 2, 8, 4
    sequences = torch.randn(batch_size, seq_len, features)
    lengths = torch.tensor([6, 8])

    packed = pack_encoder_sequences(sequences, lengths)

    # Unpack to original length
    unpacked_original, _ = unpack_to_fixed_length(packed, total_length=seq_len)
    assert unpacked_original.shape == (batch_size, seq_len, features)

    # Unpack to longer length (should pad with zeros)
    unpacked_longer, _ = unpack_to_fixed_length(packed, total_length=12)
    assert unpacked_longer.shape == (batch_size, 12, features)

    # Test default unpacking (no total_length specified)
    unpacked_default, _ = unpack_to_fixed_length(packed)
    assert unpacked_default.shape[0] == batch_size
    assert unpacked_default.shape[2] == features


def test_align_sequences_max_length_parameter():
    """Test align_encoder_sequences with different max_length values."""
    sequences = torch.tensor(
        [
            [[1, 2], [3, 4], [0, 0], [0, 0]],  # length=2
            [[5, 6], [7, 8], [9, 10], [0, 0]],  # length=3
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2, 3])

    # Test with max_length smaller than sequence length
    aligned_small = align_encoder_sequences(sequences, lengths, max_length=3)
    assert aligned_small.shape == sequences.shape

    # Test with max_length equal to sequence length
    aligned_equal = align_encoder_sequences(sequences, lengths, max_length=4)
    assert aligned_equal.shape == sequences.shape

    # The alignment should work correctly regardless
    # First sequence should have padding moved to start for length=2
    expected_alignment = torch.tensor(
        [[0, 0], [0, 0], [1, 2], [3, 4]], dtype=torch.float32
    )
    torch.testing.assert_close(aligned_equal[0], expected_alignment)
