from __future__ import annotations

import pytest
import torch
import torch.nn.utils.rnn as rnn_utils

from transformertf.utils.sequence import (
    align_encoder_sequences,
    pack_decoder_sequences,
    pack_encoder_sequences,
    should_use_packing,
    unpack_to_fixed_length,
    validate_encoder_alignment,
)


def test_should_use_packing():
    """Test the decision logic for when to use packed sequences."""
    # None input should return False
    assert not should_use_packing(None)

    # Empty tensor should return False
    empty_lengths = torch.tensor([])
    assert not should_use_packing(empty_lengths)

    # Uniform lengths should return False (no benefit from packing)
    uniform_lengths = torch.tensor([10, 10, 10, 10])
    assert not should_use_packing(uniform_lengths)

    # Small variation but large batch should use packing
    small_var_large_batch = torch.tensor([10, 9, 10, 9, 10, 9, 10, 9])
    assert should_use_packing(small_var_large_batch)

    # Large variation should use packing regardless of batch size
    large_variation = torch.tensor([10, 5, 8])  # 50% variation
    assert should_use_packing(large_variation)

    # Small batch with small variation should not use packing
    small_batch_small_var = torch.tensor([10, 9, 10])  # 10% variation, batch=3
    assert not should_use_packing(small_batch_small_var)


def test_align_encoder_sequences():
    """Test encoder sequence alignment from right-padding to left-padding."""
    # Create test sequences: right-padded
    sequences = torch.tensor([
        [[1, 2], [3, 4], [0, 0]],  # length=2, padding at end
        [[5, 6], [7, 8], [9, 10]],  # length=3, no padding
        [[11, 12], [0, 0], [0, 0]],  # length=1, padding at end
    ])
    lengths = torch.tensor([2, 3, 1])

    aligned = align_encoder_sequences(sequences, lengths)

    # Expected: left-padded (padding at beginning)
    expected = torch.tensor([
        [[0, 0], [1, 2], [3, 4]],  # padding moved to beginning
        [[5, 6], [7, 8], [9, 10]],  # no change needed
        [[0, 0], [0, 0], [11, 12]],  # padding moved to beginning
    ])

    assert torch.equal(aligned, expected)


def test_pack_encoder_sequences():
    """Test encoder sequence packing with alignment."""
    sequences = torch.tensor([
        [[1, 2], [3, 4], [0, 0]],  # length=2
        [[5, 6], [7, 8], [9, 10]],  # length=3
    ])
    lengths = torch.tensor([2, 3])

    # Test with alignment
    packed = pack_encoder_sequences(sequences, lengths, align_first=True)
    assert isinstance(packed, rnn_utils.PackedSequence)

    # Test without alignment (sequences already aligned)
    aligned_sequences = align_encoder_sequences(sequences, lengths)
    packed_no_align = pack_encoder_sequences(
        aligned_sequences, lengths, align_first=False
    )
    assert isinstance(packed_no_align, rnn_utils.PackedSequence)


def test_pack_decoder_sequences():
    """Test decoder sequence packing (already right-padded)."""
    sequences = torch.tensor([
        [[1, 2], [3, 4], [0, 0]],  # length=2
        [[5, 6], [7, 8], [9, 10]],  # length=3
    ])
    lengths = torch.tensor([2, 3])

    packed = pack_decoder_sequences(sequences, lengths)
    assert isinstance(packed, rnn_utils.PackedSequence)


def test_unpack_to_fixed_length():
    """Test unpacking sequences to consistent tensor format."""
    # Create packed sequence
    sequences = torch.tensor([
        [[1, 2], [3, 4]],  # length=2
        [[5, 6], [7, 8]],  # length=2
    ])
    lengths = torch.tensor([2, 2])

    packed = rnn_utils.pack_padded_sequence(sequences, lengths, batch_first=True)

    # Unpack
    unpacked_seqs, unpacked_lengths = unpack_to_fixed_length(packed)

    assert unpacked_seqs.shape == sequences.shape
    assert torch.equal(unpacked_lengths, lengths)


def test_encoder_alignment_validation():
    """Test encoder alignment validation using actual tensor structure."""
    # Test left alignment (padding at start)
    left_aligned = torch.tensor([
        [[0, 0], [1, 2], [3, 4]],  # length=2, padding at start
        [[0, 0], [0, 0], [5, 6]],  # length=1, padding at start
    ])
    lengths = torch.tensor([2, 1])

    # Should pass for left alignment
    validate_encoder_alignment(left_aligned, lengths, "left")

    # Should fail for right alignment
    with pytest.raises(ValueError, match="Expected right alignment.*padding at end"):
        validate_encoder_alignment(left_aligned, lengths, "right")

    # Test right alignment (padding at end)
    right_aligned = torch.tensor([
        [[1, 2], [3, 4], [0, 0]],  # length=2, padding at end
        [[5, 6], [0, 0], [0, 0]],  # length=1, padding at end
    ])

    # Should pass for right alignment
    validate_encoder_alignment(right_aligned, lengths, "right")

    # Should fail for left alignment
    with pytest.raises(ValueError, match="Expected left alignment.*padding at start"):
        validate_encoder_alignment(right_aligned, lengths, "left")

    # Test with no length information (should pass)
    validate_encoder_alignment(left_aligned, None, "left")
    validate_encoder_alignment(right_aligned, None, "right")


def test_alignment_with_variable_max_length():
    """Test alignment behavior with different max lengths."""
    sequences = torch.tensor([
        [[1, 2], [3, 4], [0, 0], [0, 0]],  # 4 timesteps
        [[5, 6], [7, 8], [9, 10], [0, 0]],  # 4 timesteps
    ])
    lengths = torch.tensor([2, 3])

    # Test with custom max length
    aligned = align_encoder_sequences(sequences, lengths, max_length=4)

    expected = torch.tensor([
        [[0, 0], [0, 0], [1, 2], [3, 4]],  # 2 padding + 2 data
        [[0, 0], [5, 6], [7, 8], [9, 10]],  # 1 padding + 3 data
    ])

    assert torch.equal(aligned, expected)


def test_zero_length_sequences():
    """Test handling of edge cases like zero-length sequences."""
    sequences = torch.tensor([
        [[1, 2], [3, 4], [0, 0]],
        [[5, 6], [0, 0], [0, 0]],
    ])
    lengths = torch.tensor([2, 0])  # Second sequence has zero length

    aligned = align_encoder_sequences(sequences, lengths)

    # Second sequence should be all zeros (no data to move)
    assert torch.equal(aligned[1], torch.zeros_like(aligned[1]))
    # First sequence should be properly aligned
    expected_first = torch.tensor([[0, 0], [1, 2], [3, 4]])
    assert torch.equal(aligned[0], expected_first)


def test_encoder_alignment_edge_cases():
    """Test edge cases for encoder alignment validation."""
    # Test sequences where length equals sequence length (no padding)
    full_length_sequences = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],  # length=3, no padding
        [[7, 8], [9, 10], [11, 12]],  # length=3, no padding
    ])
    full_lengths = torch.tensor([3, 3])

    # Should pass for both alignments when no padding exists
    validate_encoder_alignment(full_length_sequences, full_lengths, "left")
    validate_encoder_alignment(full_length_sequences, full_lengths, "right")

    # Test with mixed length sequences
    mixed_sequences = torch.tensor([
        [[0, 0], [1, 2], [3, 4]],  # length=2, left-aligned
        [[5, 6], [7, 8], [9, 10]],  # length=3, no padding
    ])
    mixed_lengths = torch.tensor([2, 3])

    # Should pass for left alignment
    validate_encoder_alignment(mixed_sequences, mixed_lengths, "left")
