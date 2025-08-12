#!/usr/bin/env python3
"""Simple alignment demonstration without plotting."""

from __future__ import annotations

import torch
from transformertf.models._base_transformer import create_mask, get_attention_mask
from transformertf.utils.sequence import align_encoder_sequences

def main():
    """Run a simple demonstration of the alignment fix."""
    print("SEQUENCE ALIGNMENT DEMONSTRATION")
    print("=" * 50)
    
    # Create sample sequences: length [3, 2] padded to size 5
    sequences = torch.tensor([
        [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [0.0, 0.0], [0.0, 0.0]],  # len=3
        [[4.0, 0.4], [5.0, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]   # len=2
    ])
    lengths = torch.tensor([3, 2])
    
    print(f"Original sequences (left-aligned):")
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        data = seq[:length, 0].tolist()
        padding = seq[length:, 0].tolist() 
        print(f"  Seq {i+1}: data={data}, padding={padding}")
    print()
    
    # Convert to right-aligned
    right_aligned = align_encoder_sequences(sequences, lengths)
    print(f"After align_encoder_sequences (right-aligned):")
    for i, (seq, length) in enumerate(zip(right_aligned, lengths)):
        pad_len = 5 - length
        padding = seq[:pad_len, 0].tolist()
        data = seq[pad_len:, 0].tolist()
        print(f"  Seq {i+1}: padding={padding}, data={data}")
    print()
    
    # Show masking behavior
    print("MASKING BEHAVIOR")
    print("=" * 30)
    
    left_mask = create_mask(size=5, lengths=lengths, alignment="left", inverse=False)
    right_mask = create_mask(size=5, lengths=lengths, alignment="right", inverse=False)
    
    print("Left alignment masks (True = padding):")
    print(left_mask)
    print("\nRight alignment masks (True = padding):")
    print(right_mask)
    print()
    
    # Show attention masks
    print("ATTENTION MASKS")
    print("=" * 25)
    
    encoder_lengths = torch.tensor([2, 3])
    decoder_lengths = torch.tensor([2, 2])
    
    left_attn = get_attention_mask(
        encoder_lengths=encoder_lengths, 
        decoder_lengths=decoder_lengths, 
        max_encoder_length=3, 
        max_decoder_length=2, 
        causal_attention=False, 
        encoder_alignment="left", 
        decoder_alignment="left"
    )
    right_attn = get_attention_mask(
        encoder_lengths=encoder_lengths, 
        decoder_lengths=decoder_lengths, 
        max_encoder_length=3, 
        max_decoder_length=2, 
        causal_attention=False, 
        encoder_alignment="right", 
        decoder_alignment="right"
    )
    
    print(f"Left alignment attention (shape {left_attn.shape}):")
    print("Sample 1:")
    print(left_attn[0])
    print("\nRight alignment attention:")
    print("Sample 1:")
    print(right_attn[0])
    
    print("\n" + "=" * 50)
    print("KEY INSIGHT:")
    print("- Left alignment: data at start → compatible with PyTorch")
    print("- Right alignment: data at end → TFT-style models")
    print("- Fixed convention eliminates pack_padded_sequence issues")

if __name__ == "__main__":
    main()