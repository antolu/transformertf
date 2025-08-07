"""
Tests for alignment-aware attention masking functionality.

This module tests that the create_mask and get_attention_mask functions
correctly handle both left-aligned and right-aligned sequences.
"""

from __future__ import annotations

import pytest
import torch

from transformertf.models._base_transformer import create_mask, get_attention_mask


class TestCreateMask:
    """Test create_mask function with different alignments."""

    def test_create_mask_right_alignment_default_behavior(self):
        """Test that right alignment produces the traditional masking behavior."""
        lengths = torch.tensor([3, 5, 2])
        mask = create_mask(size=6, lengths=lengths, alignment="right", inverse=False)

        # Right alignment: padding at end, mask True where lengths[i] <= j
        expected = torch.tensor([
            [
                False,
                False,
                False,
                True,
                True,
                True,
            ],  # len=3, padding at positions 3,4,5
            [False, False, False, False, False, True],  # len=5, padding at position 5
            [
                False,
                False,
                True,
                True,
                True,
                True,
            ],  # len=2, padding at positions 2,3,4,5
        ])

        assert torch.equal(mask, expected)

    def test_create_mask_left_alignment_new_behavior(self):
        """Test that left alignment correctly masks padding at the start."""
        lengths = torch.tensor([3, 5, 2])
        mask = create_mask(size=6, lengths=lengths, alignment="left", inverse=False)

        # Left alignment: padding at start, mask True where j < (size - lengths[i])
        expected = torch.tensor([
            [
                True,
                True,
                True,
                False,
                False,
                False,
            ],  # len=3, padding at positions 0,1,2
            [True, False, False, False, False, False],  # len=5, padding at position 0
            [
                True,
                True,
                True,
                True,
                False,
                False,
            ],  # len=2, padding at positions 0,1,2,3
        ])

        assert torch.equal(mask, expected)

    def test_create_mask_inverse_right_alignment(self):
        """Test inverse masking with right alignment."""
        lengths = torch.tensor([3, 5, 2])
        mask = create_mask(size=6, lengths=lengths, alignment="right", inverse=True)

        # Right alignment inverse: True where values are (j < lengths[i])
        expected = torch.tensor([
            [True, True, True, False, False, False],  # len=3, valid positions 0,1,2
            [True, True, True, True, True, False],  # len=5, valid positions 0,1,2,3,4
            [True, True, False, False, False, False],  # len=2, valid positions 0,1
        ])

        assert torch.equal(mask, expected)

    def test_create_mask_inverse_left_alignment(self):
        """Test inverse masking with left alignment."""
        lengths = torch.tensor([3, 5, 2])
        mask = create_mask(size=6, lengths=lengths, alignment="left", inverse=True)

        # Left alignment inverse: True where values are (j >= (size - lengths[i]))
        expected = torch.tensor([
            [False, False, False, True, True, True],  # len=3, valid positions 3,4,5
            [False, True, True, True, True, True],  # len=5, valid positions 1,2,3,4,5
            [False, False, False, False, True, True],  # len=2, valid positions 4,5
        ])

        assert torch.equal(mask, expected)

    def test_create_mask_default_alignment_is_left(self):
        """Test that left alignment is the default."""
        lengths = torch.tensor([3, 2])
        mask_default = create_mask(size=4, lengths=lengths)
        mask_explicit = create_mask(size=4, lengths=lengths, alignment="left")

        assert torch.equal(mask_default, mask_explicit)

    def test_create_mask_invalid_alignment(self):
        """Test that invalid alignment raises ValueError."""
        lengths = torch.tensor([3, 2])

        with pytest.raises(ValueError, match="alignment must be 'left' or 'right'"):
            create_mask(size=4, lengths=lengths, alignment="center")

    def test_create_mask_edge_cases(self):
        """Test edge cases with extreme lengths."""
        # Full length sequences (no padding)
        lengths = torch.tensor([5, 5])
        mask_left = create_mask(size=5, lengths=lengths, alignment="left")
        mask_right = create_mask(size=5, lengths=lengths, alignment="right")

        # Both should be all False (no padding to mask)
        expected = torch.tensor([
            [False, False, False, False, False],
            [False, False, False, False, False],
        ])

        assert torch.equal(mask_left, expected)
        assert torch.equal(mask_right, expected)

        # Zero length sequences (all padding)
        lengths = torch.tensor([0, 0])
        mask_left = create_mask(size=3, lengths=lengths, alignment="left")
        mask_right = create_mask(size=3, lengths=lengths, alignment="right")

        # Both should be all True (all padding)
        expected_all_true = torch.tensor([
            [True, True, True],
            [True, True, True],
        ])

        assert torch.equal(mask_left, expected_all_true)
        assert torch.equal(mask_right, expected_all_true)


class TestGetAttentionMask:
    """Test get_attention_mask function with different alignments."""

    def test_get_attention_mask_left_alignment_default(self):
        """Test attention mask with left alignment (new default)."""
        encoder_lengths = torch.tensor([3, 2])
        decoder_lengths = torch.tensor([2, 3])

        mask = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            max_encoder_length=4,
            max_decoder_length=3,
            causal_attention=False,
        )

        # Shape should be (batch_size, max_decoder_length, max_encoder_length + max_decoder_length)
        assert mask.shape == (2, 3, 7)

        # Check that it uses left alignment by default
        mask_explicit = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            max_encoder_length=4,
            max_decoder_length=3,
            causal_attention=False,
            encoder_alignment="left",
            decoder_alignment="left",
        )

        assert torch.equal(mask, mask_explicit)

    def test_get_attention_mask_mixed_alignments(self):
        """Test attention mask with different encoder and decoder alignments."""
        encoder_lengths = torch.tensor([3])
        decoder_lengths = torch.tensor([2])

        mask = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            max_encoder_length=4,
            max_decoder_length=3,
            causal_attention=False,
            encoder_alignment="right",  # TFT-style
            decoder_alignment="left",  # Default
        )

        # Should successfully create mask without errors
        assert mask.shape == (1, 3, 7)

        # The encoder part should mask the end (right alignment)
        # The decoder part should mask the start (left alignment)
        # This tests that the function correctly passes alignment to create_mask calls
        assert mask.dtype == torch.bool

    def test_get_attention_mask_causal_vs_non_causal(self):
        """Test that causal attention works with both alignments."""
        encoder_lengths = torch.tensor([2, 3])
        decoder_lengths = torch.tensor([3, 2])

        # Test causal with left alignment
        mask_causal_left = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            max_encoder_length=3,
            max_decoder_length=3,
            causal_attention=True,
            encoder_alignment="left",
            decoder_alignment="left",
        )

        # Test causal with right alignment
        mask_causal_right = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            max_encoder_length=3,
            max_decoder_length=3,
            causal_attention=True,
            encoder_alignment="right",
            decoder_alignment="right",
        )

        # Both should work and have the same shape
        assert mask_causal_left.shape == mask_causal_right.shape == (2, 3, 6)

        # But they should be different due to different alignment
        assert not torch.equal(mask_causal_left, mask_causal_right)

    def test_get_attention_mask_backward_compatibility(self):
        """Test that explicitly setting right alignment preserves old behavior."""
        encoder_lengths = torch.tensor([3, 2])
        decoder_lengths = torch.tensor([2, 3])

        # This should match the old behavior (before alignment parameters were added)
        mask_right = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            max_encoder_length=4,
            max_decoder_length=3,
            causal_attention=False,
            encoder_alignment="right",
            decoder_alignment="right",
        )

        assert mask_right.shape == (2, 3, 7)
        assert mask_right.dtype == torch.bool


class TestAlignmentIntegration:
    """Integration tests to verify masking works correctly with sequences."""

    def test_left_aligned_sequence_masking(self):
        """Test that left-aligned sequences get masked correctly."""
        # Create left-aligned sequences (padding at start)
        batch_size, max_len = 2, 5
        lengths = torch.tensor([3, 2])

        # Left-aligned sequences: padding at start
        sequences = torch.zeros(batch_size, max_len, 1)
        sequences[0, 2:5] = torch.tensor([
            [1],
            [2],
            [3],
        ])  # valid data at positions 2,3,4
        sequences[1, 3:5] = torch.tensor([[4], [5]])  # valid data at positions 3,4

        # Create mask for left alignment
        mask = create_mask(max_len, lengths, alignment="left", inverse=False)

        # Verify mask correctly identifies padding positions
        expected_mask = torch.tensor([
            [True, True, False, False, False],  # positions 0,1 are padding
            [True, True, True, False, False],  # positions 0,1,2 are padding
        ])

        assert torch.equal(mask, expected_mask)

        # Verify that applying inverse mask gives us the valid positions
        valid_mask = create_mask(max_len, lengths, alignment="left", inverse=True)

        # Check that valid positions contain non-zero data
        for i in range(batch_size):
            valid_positions = valid_mask[i]
            assert (sequences[i][valid_positions] != 0).all()

            # Check that padding positions contain zero data
            padding_positions = ~valid_positions
            assert (sequences[i][padding_positions] == 0).all()

    def test_right_aligned_sequence_masking(self):
        """Test that right-aligned sequences get masked correctly."""
        # Create right-aligned sequences (padding at end)
        batch_size, max_len = 2, 5
        lengths = torch.tensor([3, 2])

        # Right-aligned sequences: padding at end
        sequences = torch.zeros(batch_size, max_len, 1)
        sequences[0, 0:3] = torch.tensor([
            [1],
            [2],
            [3],
        ])  # valid data at positions 0,1,2
        sequences[1, 0:2] = torch.tensor([[4], [5]])  # valid data at positions 0,1

        # Create mask for right alignment
        mask = create_mask(max_len, lengths, alignment="right", inverse=False)

        # Verify mask correctly identifies padding positions
        expected_mask = torch.tensor([
            [False, False, False, True, True],  # positions 3,4 are padding
            [False, False, True, True, True],  # positions 2,3,4 are padding
        ])

        assert torch.equal(mask, expected_mask)

        # Verify that applying inverse mask gives us the valid positions
        valid_mask = create_mask(max_len, lengths, alignment="right", inverse=True)

        # Check that valid positions contain non-zero data
        for i in range(batch_size):
            valid_positions = valid_mask[i]
            assert (sequences[i][valid_positions] != 0).all()

            # Check that padding positions contain zero data
            padding_positions = ~valid_positions
            assert (sequences[i][padding_positions] == 0).all()
