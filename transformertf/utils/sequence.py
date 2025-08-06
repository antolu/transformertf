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


def validate_encoder_alignment(model_class_name: str, encoder_alignment: str) -> None:
    """
    Validate that the encoder alignment is compatible with the model type.

    This function checks if the specified encoder alignment is appropriate for
    the given model class and raises an informative error if there's a mismatch.

    Parameters
    ----------
    model_class_name : str
        The name of the model class to validate against.
    encoder_alignment : str
        The encoder alignment setting ("left" or "right").

    Raises
    ------
    ValueError
        If the encoder alignment is incompatible with the model type.

    Examples
    --------
    >>> validate_encoder_alignment("AttentionLSTMModel", "left")  # OK
    >>> validate_encoder_alignment("TemporalFusionTransformerModel", "right")  # OK
    >>> validate_encoder_alignment("AttentionLSTMModel", "right")  # Raises ValueError
    """
    # LSTM models require left alignment for efficient packing
    lstm_models = {"AttentionLSTMModel", "EncoderDecoderLSTM", "LSTM"}

    # TFT models require right alignment for backwards compatibility
    tft_models = {"TemporalFusionTransformerModel", "TemporalFusionTransformer"}

    if model_class_name in lstm_models and encoder_alignment != "left":
        msg = (
            f"Model '{model_class_name}' requires encoder_alignment='left' for efficient "
            f"packed sequence processing, but got '{encoder_alignment}'. "
            f"Please update your DataModule configuration to use encoder_alignment='left'."
        )
        raise ValueError(msg)

    if model_class_name in tft_models and encoder_alignment != "right":
        msg = (
            f"Model '{model_class_name}' requires encoder_alignment='right' for backwards "
            f"compatibility, but got '{encoder_alignment}'. "
            f"Please update your DataModule configuration to use encoder_alignment='right', "
            f"or use the migration script to update your model configuration."
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

    # Check if all lengths are the same
    min_len = lengths.min()
    max_len = lengths.max()

    # No variation in lengths - packing adds overhead without benefit
    if min_len == max_len:
        return False

    # Use packing if there's significant variation (>10% difference)
    # or if the batch is large enough to benefit from the efficiency
    length_variation = (max_len - min_len).float() / max_len.float()
    batch_size = lengths.numel()

    return length_variation > 0.1 or batch_size >= 8


def align_encoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    """
    Align encoder sequences from right-padding to left-padding for packing.

    When randomize_seq_len=True, encoder sequences are shortened from the left
    but then right-padded during collation. LSTM models need these sequences
    to be left-padded (aligned to the right) for efficient packing.

    Parameters
    ----------
    sequences : torch.Tensor
        Input sequences of shape (batch_size, seq_len, features).
        Assumed to be right-padded (padding at the end).
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    max_length : int, optional
        Maximum sequence length. If None, uses sequences.size(1).

    Returns
    -------
    torch.Tensor
        Sequences aligned for packing, with padding moved to the beginning.

    Examples
    --------
    >>> sequences = torch.tensor([[[1, 2], [3, 4], [0, 0]],  # len=2
    ...                          [[5, 6], [7, 8], [9, 10]]])  # len=3
    >>> lengths = torch.tensor([2, 3])
    >>> aligned = align_encoder_sequences(sequences, lengths)
    >>> # Result: [[[0, 0], [1, 2], [3, 4]],
    >>> #          [[5, 6], [7, 8], [9, 10]]]
    """
    _batch_size, seq_len, _num_features = sequences.shape
    max_length = max_length or seq_len

    # Create output tensor
    aligned = torch.zeros_like(sequences)

    for i, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            continue

        # Calculate padding amount (how much to shift right)
        padding_amount = max_length - length

        # Copy the valid sequence data to the right position
        if padding_amount > 0:
            # Move non-padded data to the right
            aligned[i, padding_amount : padding_amount + length] = sequences[i, :length]
        else:
            # No padding needed
            aligned[i] = sequences[i]

    return aligned


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

    This function handles the alignment from right-padding to left-padding
    and then packs the sequences for RNN efficiency.

    Parameters
    ----------
    sequences : torch.Tensor
        Input sequences of shape (batch_size, seq_len, features).
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    align_first : bool, default=True
        Whether to align sequences before packing (move padding to beginning).
    batch_first : bool, default=True
        Whether input has batch dimension first.
    enforce_sorted : bool, default=False
        Whether to sort sequences by length (can be more efficient).

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


def pack_decoder_sequences(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    *,
    batch_first: bool = True,
    enforce_sorted: bool = False,
) -> rnn_utils.PackedSequence:
    """
    Pack decoder sequences for efficient LSTM processing.

    Decoder sequences are already right-padded (padding at the end) which
    is the expected format for packing, so no alignment is needed.

    Parameters
    ----------
    sequences : torch.Tensor
        Input sequences of shape (batch_size, seq_len, features).
        Expected to be right-padded.
    lengths : torch.Tensor
        Actual lengths of each sequence, shape (batch_size,).
    batch_first : bool, default=True
        Whether input has batch dimension first.
    enforce_sorted : bool, default=False
        Whether to sort sequences by length (can be more efficient).

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
