#!/usr/bin/env python3
"""Create visual plots of the alignment demonstration."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformertf.models._base_transformer import create_mask
from transformertf.utils.sequence import align_encoder_sequences

def main():
    """Create alignment visualization plots."""
    # Create sample data
    sequences = torch.tensor([
        [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [0.0, 0.0], [0.0, 0.0]],  # len=3
        [[4.0, 0.4], [5.0, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # len=2
        [[6.0, 0.6], [7.0, 0.7], [8.0, 0.8], [9.0, 0.9], [0.0, 0.0]]   # len=4
    ])
    lengths = torch.tensor([3, 2, 4])
    
    # Get aligned sequences
    right_aligned = align_encoder_sequences(sequences, lengths)
    
    # Get masks
    left_mask = create_mask(size=5, lengths=lengths, alignment="left", inverse=False)
    right_mask = create_mask(size=5, lengths=lengths, alignment="right", inverse=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Sequence Alignment Convention Fix Demonstration', fontsize=14, fontweight='bold')
    
    # Plot 1: Original sequences (left-aligned)
    ax = axes[0, 0]
    seq_data = sequences[:, :, 0].numpy()  # First feature only
    im1 = ax.imshow(seq_data, cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_title('Original Sequences\n(Left-aligned: data at start)', fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    
    # Add annotations
    for i, length in enumerate(lengths):
        ax.axvline(length - 0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.text(length + 0.1, i, f'len={length}', color='red', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Color bar
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('Values', rotation=270, labelpad=15)
    
    # Plot 2: Right-aligned sequences  
    ax = axes[0, 1]
    right_data = right_aligned[:, :, 0].numpy()
    im2 = ax.imshow(right_data, cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_title('After align_encoder_sequences\n(Right-aligned: data at end)', fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    
    # Add annotations
    for i, length in enumerate(lengths):
        start_pos = 5 - length
        ax.axvline(start_pos - 0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.text(start_pos - 0.4, i, f'len={length}', color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Color bar
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('Values', rotation=270, labelpad=15)
    
    # Plot 3: Left alignment mask
    ax = axes[1, 0]
    left_mask_data = left_mask.numpy().astype(float)
    im3 = ax.imshow(left_mask_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax.set_title('Left Alignment Mask\n(Red = padding, Blue = data)', fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    
    # Add text annotations
    for i in range(left_mask_data.shape[0]):
        for j in range(left_mask_data.shape[1]):
            text = 'PAD' if left_mask_data[i, j] else 'DATA'
            color = 'white' if left_mask_data[i, j] else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold', fontsize=8)
    
    # Color bar
    cbar3 = plt.colorbar(im3, ax=ax)
    cbar3.set_label('Mask (1=pad, 0=data)', rotation=270, labelpad=15)
    
    # Plot 4: Right alignment mask
    ax = axes[1, 1] 
    right_mask_data = right_mask.numpy().astype(float)
    im4 = ax.imshow(right_mask_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax.set_title('Right Alignment Mask\n(Red = padding, Blue = data)', fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sequence Index')
    
    # Add text annotations
    for i in range(right_mask_data.shape[0]):
        for j in range(right_mask_data.shape[1]):
            text = 'PAD' if right_mask_data[i, j] else 'DATA'
            color = 'white' if right_mask_data[i, j] else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold', fontsize=8)
    
    # Color bar
    cbar4 = plt.colorbar(im4, ax=ax)  
    cbar4.set_label('Mask (1=pad, 0=data)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Add text summary
    fig.text(0.5, 0.02, 
             'KEY: Left alignment (data at start) is PyTorch compatible. ' +
             'Right alignment (data at end) is used by TFT-style models.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('/Users/antonlu/code/cern/transformertf/alignment_visualization.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("Visualization saved to: alignment_visualization.png")
    
    # Print numerical summary
    print("\nNUMERICAL SUMMARY:")
    print("=" * 40)
    print("Sequence lengths:", lengths.tolist())
    print("\nOriginal (left-aligned) - first feature values:")
    for i, seq in enumerate(sequences):
        values = [f"{x:.1f}" if x != 0 else "0.0" for x in seq[:, 0].tolist()]
        print(f"  Seq {i+1}: {values}")
    
    print("\nRight-aligned - first feature values:")
    for i, seq in enumerate(right_aligned):
        values = [f"{x:.1f}" if x != 0 else "0.0" for x in seq[:, 0].tolist()]
        print(f"  Seq {i+1}: {values}")
        
    print("\nMask comparison (True = padding):")
    print("Left alignment masks:")
    for i, mask_row in enumerate(left_mask):
        print(f"  Seq {i+1}: {mask_row.tolist()}")
    print("Right alignment masks:")
    for i, mask_row in enumerate(right_mask):
        print(f"  Seq {i+1}: {mask_row.tolist()}")

if __name__ == "__main__":
    main()