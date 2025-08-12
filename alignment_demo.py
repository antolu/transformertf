#!/usr/bin/env python3
"""
Demonstration of alignment convention fix and attention masking.

This script shows:
1. How sequence alignment works with the new convention
2. Attention mask generation for different alignments
3. Visual comparison of the results
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformertf.models._base_transformer import create_mask, get_attention_mask
from transformertf.utils.sequence import align_encoder_sequences

def create_sample_sequences():
    """Create sample sequences with different lengths for demonstration."""
    # Create 3 sequences with lengths [3, 5, 2] padded to max length 6
    sequences = torch.tensor([
        # Sequence 1: length=3, left-aligned (data at start, padding at end)
        [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        
        # Sequence 2: length=5, left-aligned (data at start, padding at end)  
        [[4.0, 0.4], [5.0, 0.5], [6.0, 0.6], [7.0, 0.7], [8.0, 0.8], [0.0, 0.0]],
        
        # Sequence 3: length=2, left-aligned (data at start, padding at end)
        [[9.0, 0.9], [10.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    ])
    
    lengths = torch.tensor([3, 5, 2])
    return sequences, lengths

def demonstrate_alignment():
    """Show how sequence alignment works with the new convention."""
    sequences, lengths = create_sample_sequences()
    
    print("=== SEQUENCE ALIGNMENT DEMONSTRATION ===")
    print(f"Original sequences (left-aligned): shape {sequences.shape}")
    print(f"Sequence lengths: {lengths.tolist()}")
    print()
    
    # Show original sequences
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        print(f"Sequence {i+1} (length={length}):")
        data_part = seq[:length, 0].tolist()  # Just first feature for clarity
        pad_part = seq[length:, 0].tolist()
        print(f"  Data: {data_part}")
        print(f"  Padding: {pad_part}")
        print()
    
    # Apply right alignment
    right_aligned = align_encoder_sequences(sequences, lengths)
    
    print("After align_encoder_sequences (converts to right-aligned):")
    for i, (seq, length) in enumerate(zip(right_aligned, lengths)):
        pad_len = 6 - length
        pad_part = seq[:pad_len, 0].tolist()
        data_part = seq[pad_len:, 0].tolist()  
        print(f"Sequence {i+1} (length={length}):")
        print(f"  Padding: {pad_part}")
        print(f"  Data: {data_part}")
        print()
    
    return sequences, right_aligned, lengths

def demonstrate_masking():
    """Show masking behavior for different alignments."""
    lengths = torch.tensor([3, 5, 2])
    size = 6
    
    print("=== ATTENTION MASKING DEMONSTRATION ===")
    print(f"Sequence lengths: {lengths.tolist()}, max_length: {size}")
    print()
    
    # Left alignment masking (padding at end)
    left_mask = create_mask(size, lengths, alignment="left", inverse=False)
    left_valid = create_mask(size, lengths, alignment="left", inverse=True)
    
    print("LEFT ALIGNMENT (data at start, padding at end):")
    print("Padding mask (True = padding position):")
    print(left_mask)
    print("Valid mask (True = data position):")
    print(left_valid)
    print()
    
    # Right alignment masking (padding at start)
    right_mask = create_mask(size, lengths, alignment="right", inverse=False)
    right_valid = create_mask(size, lengths, alignment="right", inverse=True)
    
    print("RIGHT ALIGNMENT (data at end, padding at start):")
    print("Padding mask (True = padding position):")
    print(right_mask)
    print("Valid mask (True = data position):")
    print(right_valid)
    print()
    
    return {
        'left_mask': left_mask,
        'left_valid': left_valid,
        'right_mask': right_mask,
        'right_valid': right_valid,
        'lengths': lengths
    }

def demonstrate_attention_masks():
    """Show attention mask generation for encoder-decoder."""
    encoder_lengths = torch.tensor([3, 2])  # 2 sequences
    decoder_lengths = torch.tensor([2, 3])
    max_encoder_length = 4
    max_decoder_length = 3
    
    print("=== ENCODER-DECODER ATTENTION MASKS ===")
    print(f"Encoder lengths: {encoder_lengths.tolist()}, max: {max_encoder_length}")
    print(f"Decoder lengths: {decoder_lengths.tolist()}, max: {max_decoder_length}")
    print()
    
    # Left alignment (default)
    left_attn_mask = get_attention_mask(
        encoder_lengths=encoder_lengths,
        decoder_lengths=decoder_lengths,
        max_encoder_length=max_encoder_length,
        max_decoder_length=max_decoder_length,
        causal_attention=False,
        encoder_alignment="left",
        decoder_alignment="left"
    )
    
    print("Left alignment attention mask (False = can attend):")
    print(f"Shape: {left_attn_mask.shape} (batch, decoder_seq, encoder_seq + decoder_seq)")
    for i in range(left_attn_mask.shape[0]):
        print(f"Sample {i+1}:")
        print(left_attn_mask[i])
        print()
    
    # Right alignment  
    right_attn_mask = get_attention_mask(
        encoder_lengths=encoder_lengths,
        decoder_lengths=decoder_lengths,
        max_encoder_length=max_encoder_length,
        max_decoder_length=max_decoder_length,
        causal_attention=False,
        encoder_alignment="right",
        decoder_alignment="right"
    )
    
    print("Right alignment attention mask (False = can attend):")
    print(f"Shape: {right_attn_mask.shape}")
    for i in range(right_attn_mask.shape[0]):
        print(f"Sample {i+1}:")
        print(right_attn_mask[i])
        print()
    
    return left_attn_mask, right_attn_mask

def plot_results():
    """Create visualizations of alignment and masking."""
    sequences, right_aligned, lengths = demonstrate_alignment()
    mask_data = demonstrate_masking()
    left_attn, right_attn = demonstrate_attention_masks()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sequence Alignment and Masking Demonstration', fontsize=16)
    
    # Plot 1: Original sequences (left-aligned)
    ax = axes[0, 0]
    seq_vis = sequences[:, :, 0].numpy()  # Just first feature
    im1 = ax.imshow(seq_vis, cmap='Blues', aspect='auto')
    ax.set_title('Original Sequences\n(Left-aligned: data at start)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    
    # Add length annotations
    for i, length in enumerate(lengths):
        ax.axvline(length - 0.5, color='red', linestyle='--', alpha=0.7)
        ax.text(length, i, f'len={length}', color='red', fontweight='bold')
    
    plt.colorbar(im1, ax=ax)
    
    # Plot 2: Right-aligned sequences
    ax = axes[0, 1]
    right_vis = right_aligned[:, :, 0].numpy()
    im2 = ax.imshow(right_vis, cmap='Blues', aspect='auto')
    ax.set_title('After Alignment\n(Right-aligned: data at end)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    
    # Add length annotations
    for i, length in enumerate(lengths):
        start_pos = 6 - length
        ax.axvline(start_pos - 0.5, color='red', linestyle='--', alpha=0.7)
        ax.text(start_pos, i, f'len={length}', color='red', fontweight='bold')
    
    plt.colorbar(im2, ax=ax)
    
    # Plot 3: Left alignment masks
    ax = axes[0, 2]
    left_combined = np.stack([
        mask_data['left_mask'].numpy().astype(float),
        mask_data['left_valid'].numpy().astype(float) * 0.5
    ], axis=-1).sum(axis=-1)
    im3 = ax.imshow(left_combined, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Left Alignment Masks\n(Dark = padding, Light = data)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    plt.colorbar(im3, ax=ax)
    
    # Plot 4: Right alignment masks  
    ax = axes[1, 0]
    right_combined = np.stack([
        mask_data['right_mask'].numpy().astype(float),
        mask_data['right_valid'].numpy().astype(float) * 0.5
    ], axis=-1).sum(axis=-1)
    im4 = ax.imshow(right_combined, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Right Alignment Masks\n(Dark = padding, Light = data)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    plt.colorbar(im4, ax=ax)
    
    # Plot 5: Left attention mask (first sample)
    ax = axes[1, 1]
    left_attn_vis = left_attn[0].numpy().astype(float)
    im5 = ax.imshow(left_attn_vis, cmap='Reds', aspect='auto')
    ax.set_title('Left Alignment Attention\n(Red = masked, White = can attend)')
    ax.set_xlabel('Encoder + Decoder Positions')
    ax.set_ylabel('Decoder Query Positions')
    ax.axvline(3.5, color='blue', linestyle='--', alpha=0.7)  # Encoder-decoder boundary
    ax.text(1.5, -0.3, 'Encoder', ha='center', color='blue', fontweight='bold')
    ax.text(5.5, -0.3, 'Decoder', ha='center', color='blue', fontweight='bold')
    plt.colorbar(im5, ax=ax)
    
    # Plot 6: Right attention mask (first sample)
    ax = axes[1, 2]
    right_attn_vis = right_attn[0].numpy().astype(float)
    im6 = ax.imshow(right_attn_vis, cmap='Reds', aspect='auto')
    ax.set_title('Right Alignment Attention\n(Red = masked, White = can attend)')
    ax.set_xlabel('Encoder + Decoder Positions')  
    ax.set_ylabel('Decoder Query Positions')
    ax.axvline(3.5, color='blue', linestyle='--', alpha=0.7)  # Encoder-decoder boundary
    ax.text(1.5, -0.3, 'Encoder', ha='center', color='blue', fontweight='bold')
    ax.text(5.5, -0.3, 'Decoder', ha='center', color='blue', fontweight='bold')
    plt.colorbar(im6, ax=ax)
    
    plt.tight_layout()
    plt.savefig('/Users/antonlu/code/cern/transformertf/alignment_demo.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: /Users/antonlu/code/cern/transformertf/alignment_demo.png")
    plt.show()

def main():
    """Run the complete demonstration."""
    print("TRANSFORMERTF ALIGNMENT CONVENTION FIX DEMONSTRATION")
    print("=" * 60)
    print()
    
    demonstrate_alignment()
    demonstrate_masking()  
    demonstrate_attention_masks()
    plot_results()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- Left alignment: data at start, padding at end (PyTorch compatible)")
    print("- Right alignment: data at end, padding at start (TFT-style)")
    print("- Attention masks correctly handle both alignment strategies")
    print("- The fix ensures PyTorch pack_padded_sequence compatibility")

if __name__ == "__main__":
    main()