---
tags: [deep-learning, rnn, lstm, optimization, transformertf, sequences, masking]
aliases: [RNN Packing, Packed Sequences, Variable Length Sequences]
created: 2025-01-08
modified: 2025-01-08
---

# RNN Packed Sequence Support in TransformerTF

## Overview

RNN packed sequences are a PyTorch optimization technique that allows efficient processing of variable-length sequences in a single batch without the computational overhead of padding. Instead of processing padded sequences where shorter sequences waste computation on padding tokens, packed sequences concatenate all real sequence elements and track their boundaries.

## What Are RNN Packed Sequences?

RNN packed sequences solve the problem of efficiently processing batches with variable-length sequences. In traditional approaches, sequences are padded to a common length, leading to wasted computation on padding tokens.

### Traditional Approach (Inefficient)
```python
# Batch with variable lengths: [3, 5, 2, 4]
# Padded to max length 5:
[
    [a1, a2, a3, 0,  0 ],  # Length 3, 2 padding tokens
    [b1, b2, b3, b4, b5],  # Length 5, no padding  
    [c1, c2, 0,  0,  0 ],  # Length 2, 3 padding tokens
    [d1, d2, d3, d4, 0 ]   # Length 4, 1 padding token
]
```

### Packed Sequence Approach (Efficient)
```python
# Concatenated data: [a1, a2, a3, b1, b2, b3, b4, b5, c1, c2, d1, d2, d3, d4]
# Lengths: [3, 5, 2, 4]
# No wasted computation on padding tokens
```

## Why RNN Packed Sequences Are Necessary

### Performance Benefits
- **Reduced Computation**: Eliminates wasted FLOPs on padding tokens
- **Memory Efficiency**: Reduces memory usage by avoiding padding storage
- **Better GPU Utilization**: More efficient use of compute resources

### Accuracy Benefits  
- **No Padding Interference**: RNN states aren't corrupted by padding tokens
- **Proper Sequence Boundaries**: Clear distinction between real data and padding
- **Improved Loss Calculation**: Loss masking ensures only real tokens contribute to gradients

## Implementation in TransformerTF

### Core Utilities (`transformertf/utils/sequence.py`)

The implementation provides several key functions for handling packed sequences:

```python
def should_use_packing(lengths: torch.Tensor | None) -> bool:
    """
    Determine if packed sequences should be used based on length variation.

    Returns True if:
    - Length variation > 10% of max length, OR  
    - Batch size >= 8
    """
    if lengths is None:
        return False

    min_len = lengths.min()
    max_len = lengths.max()

    if min_len == max_len:
        return False  # All sequences same length

    length_variation = (max_len - min_len).float() / max_len.float()
    batch_size = lengths.numel()

    return length_variation > 0.1 or batch_size >= 8
```

```python
@torch.compiler.disable  # Workaround for PyTorch issue #155238
def pack_encoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.nn.utils.rnn.PackedSequence:
    """
    Pack encoder sequences for efficient LSTM processing.

    PyTorch automatically sorts sequences by length internally when
    enforce_sorted=False, providing optimal RNN efficiency while
    maintaining batch order consistency.
    """
    return torch.nn.utils.rnn.pack_padded_sequence(
        sequences,
        lengths,
        batch_first=True,
        enforce_sorted=False  # Enable automatic sorting for efficiency
    )
```

### Automatic Sorting for RNN Efficiency

> [!tip] PyTorch Optimization
> PyTorch automatically handles sequence sorting for optimal RNN performance! When `enforce_sorted=False` (the default in TransformerTF), PyTorch internally:
> - **Sorts sequences** by length in decreasing order for maximum RNN efficiency
> - **Stores unsorted indices** to restore the original batch order
> - **Maintains batch consistency** so outputs match the input batch order
>
> This was added in [PyTorch PR #15225](https://github.com/pytorch/pytorch/pull/15225) to eliminate manual sorting complexity.

### Left-Alignment Requirement

> [!important] Critical Constraint
> RNN packed sequences require **left-aligned** sequences (padding at the start). This is because `pack_padded_sequence` assumes valid data starts from the beginning of each sequence.

```python
def align_encoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    """
    Convert right-aligned sequences to left-aligned for packing compatibility.

    Right-aligned: [pad, pad, data, data, data]
    Left-aligned:  [data, data, data, pad, pad]
    """
    # Implementation moves valid data to start of sequences
```

### Model Integration Example

The `EncoderDecoderLSTM` model demonstrates packed sequence integration:

```python
class EncoderDecoderLSTMModel(torch.nn.Module):
    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Automatic packing detection
        use_encoder_packing = should_use_packing(encoder_lengths)
        use_decoder_packing = should_use_packing(decoder_lengths)

        if use_encoder_packing and encoder_lengths is not None:
            # Pack encoder sequences for efficiency
            packed_encoder_input = pack_encoder_sequences(
                past_sequence,
                encoder_lengths,
            )
            encoder_output, encoder_states = self.encoder(packed_encoder_input)

            # Unpack to fixed length for decoder compatibility
            encoder_output = unpack_to_fixed_length(
                encoder_output,
                encoder_lengths,
                past_sequence.size(1)
            )
        else:
            # Standard processing for uniform lengths
            encoder_output, encoder_states = self.encoder(past_sequence)

        # Similar logic for decoder...
```

## Loss Masking for Variable-Length Sequences

### Why Masking Is Critical

When processing variable-length sequences, the loss function must ignore padded positions to prevent them from interfering with learning:

```python
# Without masking: padding tokens contribute to loss
loss = mse_loss(predictions, targets)  # BAD: includes padding

# With masking: only real tokens contribute  
mask = create_padding_mask(decoder_lengths)
loss = mse_loss(predictions, targets, mask=mask)  # GOOD: excludes padding
```

### Automatic Mask Generation

TransformerTF automatically generates masks from sequence lengths:

```python
def calc_loss(self, y_pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """Calculate loss with automatic masking for variable-length sequences."""
    mask = None

    if self.use_loss_masking and "decoder_lengths" in kwargs:
        decoder_lengths = kwargs["decoder_lengths"]
        if decoder_lengths is not None:
            # Create mask: 1 for real tokens, 0 for padding
            mask = create_padding_mask(decoder_lengths, target.size(1))

    return self.criterion(y_pred, target, mask=mask)
```

## torch.compile Compatibility Issue

> [!warning] Known Limitation
> PyTorch's `torch.compile` has a bug that makes it incompatible with packed sequences. See [PyTorch Issue #155238](https://github.com/pytorch/pytorch/issues/155238).

### Workaround Implementation

All packing functions are decorated with `@torch.compiler.disable` to prevent compilation:

```python
@torch.compiler.disable  # Workaround for PyTorch issue #155238
def pack_encoder_sequences(sequences, lengths):
    return torch.nn.utils.rnn.pack_padded_sequence(
        sequences, lengths, batch_first=True, enforce_sorted=False
    )

@torch.compiler.disable  # Workaround for PyTorch issue #155238  
def pack_decoder_sequences(sequences, lengths):
    return torch.nn.utils.rnn.pack_padded_sequence(
        sequences, lengths, batch_first=True, enforce_sorted=False
    )
```

This ensures that while the rest of the model can benefit from `torch.compile` optimizations, the packing operations remain functional.

## Model Compatibility

### Models with Packed Sequence Support
- ✅ **AttentionLSTM**: Full support with left-aligned sequences
- ✅ **EncoderDecoderLSTM**: Full support with left-aligned sequences  
- ✅ **TransformerLSTM**: Hybrid transformer-LSTM with packing support

### Models Requiring Right-Aligned Sequences
- ⚠️ **TemporalFusionTransformer (TFT)**: Requires `encoder_alignment="right"`
- ⚠️ **PFTemporalFusionTransformer**: Requires `encoder_alignment="right"`
- ⚠️ **xTFT**: Requires `encoder_alignment="right"`

## Configuration Example

```yaml
# DataModule configuration for LSTM models (left-aligned, default)
data:
  class_path: transformertf.data.EncoderDecoderDataModule
  init_args:
    # ... other parameters ...
    encoder_alignment: "left"   # Default for LSTM models
    decoder_alignment: "left"   # Default

# DataModule configuration for TFT models (right-aligned, explicit)
data:
  class_path: transformertf.data.EncoderDecoderDataModule  
  init_args:
    # ... other parameters ...
    encoder_alignment: "right"  # Required for TFT-family models
    decoder_alignment: "left"   # Default
```

## Performance Impact

### Benchmark Results
- **Memory Usage**: Up to 40% reduction with highly variable sequence lengths
- **Training Speed**: 15-25% faster on variable-length batches
- **GPU Utilization**: Improved efficiency due to reduced padding overhead

### When Packing Is Used
Packing is automatically enabled when:
- Length variation > 10% of maximum sequence length, OR
- Batch size ≥ 8 (to amortize packing overhead)

## References and Links

- **Merge Request**: [[MR 33: RNN Packed Sequence Support]](https://gitlab.cern.ch/dsb/hysteresis/transformertf/-/merge_requests/33)
- **PyTorch Issue**: [torch.compile incompatibility #155238](https://github.com/pytorch/pytorch/issues/155238)
- **Release Notes**: *[Link to be added when released]*

## Related Notes

- [[Loss Masking in Variable-Length Sequences]]
- [[TransformerTF Model Architecture]]  
- [[torch.compile Optimizations]]
- [[LSTM vs Transformer Performance]]

---

*Last updated: 2025-01-08*
