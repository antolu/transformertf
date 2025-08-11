from __future__ import annotations

import torch
import torch.nn.utils.rnn as rnn_utils

__all__ = [
    "align_encoder_sequences",
    "pack_decoder_sequences",
    "pack_encoder_sequences",
    "should_use_packing",
    "unpack_to_fixed_length",
    "validate_encoder_alignment",
]


def validate_encoder_alignment(
    encoder_input: torch.Tensor,
    encoder_lengths: torch.Tensor | None,
    expected_alignment: str,
) -> None:
    """
    Validate that encoder sequences are aligned as expected by checking padding location.

    This function examines the actual tensor structure to determine if sequences
    are left-aligned (padding at end) or right-aligned (padding at start).

    Parameters
    ----------
    encoder_input : torch.Tensor
        Encoder input tensor of shape (batch_size, seq_len, features).
    encoder_lengths : torch.Tensor | None
        Actual lengths of encoder sequences. If None, no validation is performed.
    expected_alignment : {"left", "right"}
        Expected alignment - "left" for padding at end, "right" for padding at start.

    Raises
    ------
    ValueError
        If the actual padding location doesn't match expected alignment.

    Examples
    --------
    >>> # Left-aligned: padding at end
    >>> input_tensor = torch.tensor([[[1, 2], [3, 4], [0, 0]]])  # data, data, pad
    >>> lengths = torch.tensor([2])
    >>> validate_encoder_alignment(input_tensor, lengths, "right")  # OK

    >>> # Right-aligned: padding at start
    >>> input_tensor = torch.tensor([[[0, 0], [1, 2], [3, 4]]])  # pad, data, data
    >>> lengths = torch.tensor([2])
    >>> validate_encoder_alignment(input_tensor, lengths, "left")  # OK
    """
    if encoder_lengths is None:
        return

    _batch_size, seq_len, _ = encoder_input.shape

    for i, length in enumerate(encoder_lengths):
        length = int(length)
        if length >= seq_len:
            # No validation needed - sequence uses full length (no padding)
            continue

        # Check if padding is at the beginning (right alignment - data at right, padding at left)
        padding_at_start = torch.allclose(
            encoder_input[i, : seq_len - length],
            torch.zeros_like(encoder_input[i, : seq_len - length]),
        )
        # Check if padding is at the end (left alignment - data at left, padding at right)
        padding_at_end = torch.allclose(
            encoder_input[i, length:], torch.zeros_like(encoder_input[i, length:])
        )

        if expected_alignment == "left" and not padding_at_end:
            msg = (
                f"Expected left alignment (data at left, padding at right) but found non-zero values at the end of sequence {i}. "
                f"Sequence length: {length}, but positions {length}: should be zero."
            )
            raise ValueError(msg)
        if expected_alignment == "right" and not padding_at_start:
            msg = (
                f"Expected right alignment (data at right, padding at left) but found non-zero values at the beginning of sequence {i}. "
                f"Sequence length: {length}, but first {seq_len - length} positions should be zero."
            )
            raise ValueError(msg)


def should_use_packing(lengths: torch.Tensor | None) -> bool:
    """
    Determine if RNN packed sequences should be used based on sequence lengths.

    Packing is beneficial when sequences have variable lengths, but adds overhead
    for uniform lengths. This function implements heuristics to decide when
    packing improves efficiency.

    Parameters
    ----------
    lengths : torch.Tensor or None
        Tensor of shape (batch_size,) containing actual sequence lengths.
        If None, assumes all sequences are the same length.

    Returns
    -------
    bool
        True if packed sequences should be used for efficiency.

    Examples
    --------
    >>> # Variable length sequences benefit from packing
    >>> lengths = torch.tensor([10, 8, 12, 6])
    >>> should_use_packing(lengths)
    True

    >>> # Uniform sequences don't need packing
    >>> lengths = torch.tensor([10, 10, 10, 10])
    >>> should_use_packing(lengths)
    False

    >>> # No length information assumes uniform
    >>> should_use_packing(None)
    False
    """
    if lengths is None:
        return False

    if lengths.numel() == 0:
        return False

    min_len = lengths.min()
    max_len = lengths.max()

    if min_len == max_len:
        return False

    length_variation = (max_len - min_len).float() / max_len.float()
    batch_size = lengths.numel()

    return bool(length_variation > 0.1 or batch_size >= 8)


def align_encoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    """
    Align encoder sequences to right-aligned format (data at right, padding at left).

    This function converts left-aligned sequences (data at left, padding at right)
    to right-aligned sequences (data at right, padding at left) for models that
    require right-aligned input data.

    Parameters
    ----------
    sequences : torch.Tensor
        Input sequences of shape (batch_size, seq_len, features).
        Assumed to be left-aligned (data at left, padding at right).
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    max_length : int, optional
        Maximum sequence length. If None, uses sequences.size(1).

    Returns
    -------
    torch.Tensor
        Right-aligned sequences with data moved to the right and padding at the left.

    Examples
    --------
    >>> # Left-aligned: data at left, padding at right
    >>> sequences = torch.tensor([[[1, 2], [3, 4], [0, 0]],  # len=2, left-aligned
    ...                          [[5, 6], [7, 8], [9, 10]]])  # len=3, left-aligned
    >>> lengths = torch.tensor([2, 3])
    >>> aligned = align_encoder_sequences(sequences, lengths)
    >>> # Result: [[[0, 0], [1, 2], [3, 4]],    # right-aligned
    >>> #          [[5, 6], [7, 8], [9, 10]]]   # right-aligned
    """
    _batch_size, seq_len, _num_features = sequences.shape
    max_length = max_length or seq_len

    aligned = torch.zeros_like(sequences)

    for i, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            continue

        padding_amount = max_length - length

        if padding_amount > 0:
            aligned[i, padding_amount : padding_amount + length] = sequences[i, :length]
        else:
            aligned[i] = sequences[i]

    return aligned


@torch.compiler.disable
def pack_encoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    *,
    align_first: bool = True,
    batch_first: bool = True,
    enforce_sorted: bool = False,
) -> rnn_utils.PackedSequence:
    """
    Pack encoder sequences for efficient LSTM processing.

    This function handles the alignment from left-padding to right-padding
    and then packs the sequences for RNN efficiency. PyTorch automatically
    sorts sequences by length internally when enforce_sorted=False, providing
    optimal RNN performance while maintaining batch order consistency.

    This function is decorated with @torch.compiler.disable to avoid
    torch.compile incompatibility with pack_padded_sequence operations.

    See PyTorch issue: https://github.com/pytorch/pytorch/issues/155238
    The issue causes Dynamo to fail during FX graph tracing when fake tensors
    encounter pack_padded_sequence and pad_packed_sequence operations.

    Parameters
    ----------
    sequences : torch.Tensor
        Input sequences of shape (batch_size, seq_len, features).
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    align_first : bool, default=True
        Whether to align sequences before packing (move data to right, padding to left).
    batch_first : bool, default=True
        Whether input has batch dimension first.
    enforce_sorted : bool, default=False
        Whether sequences are pre-sorted by length. If False, PyTorch will
        automatically sort sequences internally for optimal RNN efficiency
        and restore the original batch order in outputs.

    Returns
    -------
    torch.nn.utils.rnn.PackedSequence
        Packed sequences ready for LSTM processing.
    """
    if align_first:
        sequences = align_encoder_sequences(sequences, lengths)

    return rnn_utils.pack_padded_sequence(
        sequences,
        lengths.cpu(),  # pack_padded_sequence requires CPU lengths
        batch_first=batch_first,
        enforce_sorted=enforce_sorted,
    )


@torch.compiler.disable
def pack_decoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    *,
    batch_first: bool = True,
    enforce_sorted: bool = False,
) -> rnn_utils.PackedSequence:
    """
    Pack decoder sequences for efficient LSTM processing.

    Decoder sequences are already left-padded (data at left, padding at right) which
    is the expected format for packing, so no alignment is needed. PyTorch
    automatically sorts sequences by length internally when enforce_sorted=False,
    providing optimal RNN performance while maintaining batch order consistency.

    This function is decorated with @torch.compiler.disable to avoid
    torch.compile incompatibility with pack_padded_sequence operations.

    See PyTorch issue: https://github.com/pytorch/pytorch/issues/155238
    The issue causes Dynamo to fail during FX graph tracing when fake tensors
    encounter pack_padded_sequence and pad_packed_sequence operations.

    Parameters
    ----------
    sequences : torch.Tensor
        Input sequences of shape (batch_size, seq_len, features).
        Expected to be left-padded (data at left, padding at right).
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    batch_first : bool, default=True
        Whether input has batch dimension first.
    enforce_sorted : bool, default=False
        Whether sequences are pre-sorted by length. If False, PyTorch will
        automatically sort sequences internally for optimal RNN efficiency
        and restore the original batch order in outputs.

    Returns
    -------
    torch.nn.utils.rnn.PackedSequence
        Packed sequences ready for LSTM processing.
    """
    return rnn_utils.pack_padded_sequence(
        sequences,
        lengths.cpu(),  # pack_padded_sequence requires CPU lengths
        batch_first=batch_first,
        enforce_sorted=enforce_sorted,
    )


@torch.compiler.disable
def unpack_to_fixed_length(
    packed_sequence: rnn_utils.PackedSequence,
    *,
    batch_first: bool = True,
    total_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack sequences and return them with consistent tensor format.

    This function unpacks RNN outputs and returns both the sequences and
    their actual lengths in a format suitable for downstream processing.

    This function is decorated with @torch.compiler.disable to avoid
    torch.compile incompatibility with pad_packed_sequence operations.

    See PyTorch issue: https://github.com/pytorch/pytorch/issues/155238
    The issue causes Dynamo to fail during FX graph tracing when fake tensors
    encounter pack_padded_sequence and pad_packed_sequence operations.

    Parameters
    ----------
    packed_sequence : torch.nn.utils.rnn.PackedSequence
        Packed sequences from LSTM processing.
    batch_first : bool, default=True
        Whether to return sequences with batch dimension first.
    total_length : int, optional
        Pad sequences to this length. If None, uses the maximum length
        in the batch.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - sequences: Unpacked sequences of shape (batch_size, max_len, features)
        - lengths: Actual lengths of each sequence, shape (batch_size,)
    """
    sequences, lengths = rnn_utils.pad_packed_sequence(
        packed_sequence,
        batch_first=batch_first,
        total_length=total_length,
    )

    return sequences, lengths
