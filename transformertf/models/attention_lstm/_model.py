from __future__ import annotations

import typing

import torch

from ...nn import GateAddNorm, InterpretableMultiHeadAttention

__all__ = ["AttentionLSTMModel"]

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class AttentionLSTMModel(torch.nn.Module):
    """
    Attention-enhanced LSTM for sequence-to-sequence time series forecasting.

    This model combines encoder-decoder LSTM architecture with self-attention mechanism
    applied to the decoder outputs. The model includes optional gating for skip connections
    and supports dynamic sequence lengths for variable-length samples in batches.

    The architecture consists of:
    1. Encoder LSTM: Processes past feature sequences
    2. Decoder LSTM: Generates future predictions using encoder context
    3. Self-Attention: Captures long-range dependencies in decoder outputs
    4. Skip Connection: Optional gating or simple residual connection
    5. Linear Output: Single linear layer for final predictions

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
    hidden_size : int, default=128
        Hidden size used for both encoder and decoder LSTM layers.
    num_layers : int, default=2
        Number of LSTM layers used for both encoder and decoder.
    dropout : float, default=0.1
        Dropout probability applied to all components (LSTM, attention, output).
    n_heads : int, default=4
        Number of attention heads in the multi-head attention mechanism.
    use_gating : bool, default=True
        Whether to use gating mechanism for skip connections. If False, uses simple residual.
    trainable_add : bool, default=False
        Whether to use learnable gating for the residual connection.
    output_dim : int, default=1
        Output dimension of the model.

    Attributes
    ----------
    encoder : torch.nn.LSTM
        The encoder LSTM network.
    decoder : torch.nn.LSTM
        The decoder LSTM network.
    attention : InterpretableMultiHeadAttention
        Self-attention mechanism for decoder outputs.
    gate_add_norm : GateAddNorm or None
        Gated skip connection layer (if use_gating=True).
    layer_norm : torch.nn.LayerNorm or None
        Layer normalization for simple residual (if use_gating=False).
    output_head : torch.nn.Linear
        Linear layer for final output projection.

    Examples
    --------
    >>> import torch
    >>> from transformertf.models.attention_lstm import AttentionLSTM
    >>>
    >>> # Create model with shared parameters
    >>> model = AttentionLSTM(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     n_heads=4,
    ...     use_gating=True
    ... )
    >>>
    >>> # Forward pass with dynamic lengths
    >>> batch_size, past_len, future_len = 32, 100, 50
    >>> past_seq = torch.randn(batch_size, past_len, 10)
    >>> future_seq = torch.randn(batch_size, future_len, 5)
    >>> decoder_lengths = torch.randint(30, future_len+1, (batch_size,))
    >>> output = model(past_seq, future_seq, decoder_lengths)
    >>> print(output.shape)  # torch.Size([32, 50, 1])

    Notes
    -----
    **Shared Parameters:**

    Both encoder and decoder use the same hidden_size and num_layers to reduce
    hyperparameter complexity and ensure consistent capacity.

    **Dynamic Length Support:**

    The model handles variable sequence lengths through attention masking based
    on decoder_lengths. Padded positions are masked out during attention computation.

    **Skip Connection:**

    The model supports either gated skip connections (using GateAddNorm) or simple
    residual connections with layer normalization, controlled by the use_gating parameter.

    **Input Requirements:**

    - past_sequence: (batch_size, past_seq_len, num_past_features)
    - future_sequence: (batch_size, future_seq_len, num_future_features)
    - decoder_lengths: (batch_size,) optional tensor of actual sequence lengths

    The model supports different sequence lengths and feature dimensions for
    maximum flexibility in various forecasting scenarios.
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
        use_gating: bool = True,
        trainable_add: bool = False,
        output_dim: int = 1,
    ):
        super().__init__()

        # Validate that hidden_size is divisible by n_heads for attention mechanism
        assert hidden_size % n_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}) "
            f"for the attention mechanism to work properly."
        )

        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.use_gating = use_gating
        self.output_dim = output_dim

        # Encoder LSTM
        self.encoder = torch.nn.LSTM(
            input_size=num_past_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Decoder LSTM (same parameters as encoder)
        self.decoder = torch.nn.LSTM(
            input_size=num_future_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Self-attention mechanism
        self.attention = InterpretableMultiHeadAttention(
            n_dim_model=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Skip connection mechanism
        if use_gating:
            self.gate_add_norm = GateAddNorm(
                input_dim=hidden_size,
                dropout=dropout,
                trainable_add=trainable_add,
            )
            self.layer_norm = None
        else:
            self.gate_add_norm = None
            self.layer_norm = torch.nn.LayerNorm(hidden_size)

        # Output projection
        self.output_head = torch.nn.Linear(hidden_size, output_dim)

    def _create_attention_mask(
        self, decoder_lengths: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        """
        Create attention mask from sequence lengths.

        Parameters
        ----------
        decoder_lengths : torch.Tensor
            Tensor of shape (batch_size,) containing actual sequence lengths.
        max_length : int
            Maximum sequence length in the batch.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (max_length, max_length) where True indicates
            positions to be masked out during attention computation.
        """
        # Create mask where positions beyond sequence length are True (masked)
        # Shape: (batch_size, max_length)
        sequence_mask = (
            torch.arange(max_length, device=decoder_lengths.device)[None, :]
            >= decoder_lengths[:, None]
        )

        # For attention, we need to mask positions that are beyond the sequence length
        # in either the query or key dimension
        # Create a 2D mask: (max_length, max_length)
        # If any position i or j is beyond any sequence length, mask it
        mask = sequence_mask.any(dim=0)  # Shape: (max_length,)

        # Expand to (max_length, max_length) - mask entire rows and columns that are beyond any sequence
        return mask.unsqueeze(0) | mask.unsqueeze(1)  # Shape: (max_length, max_length)

    @typing.overload
    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
        *,
        return_encoder_states: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
        *,
        return_encoder_states: typing.Literal[True],
    ) -> tuple[torch.Tensor, HIDDEN_STATE]: ...

    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
        *,
        return_encoder_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]:
        """
        Forward pass through the encoder-decoder LSTM with self-attention.

        Parameters
        ----------
        past_sequence : torch.Tensor
            Past sequence tensor of shape (batch_size, past_seq_len, num_past_features).
        future_sequence : torch.Tensor
            Future sequence tensor of shape (batch_size, future_seq_len, num_future_features).
        encoder_lengths : torch.Tensor or None, default=None
            Tensor of shape (batch_size,) containing actual encoder sequence lengths.
            Currently not used for masking but included for API consistency.
        decoder_lengths : torch.Tensor or None, default=None
            Tensor of shape (batch_size,) containing actual decoder sequence lengths for masking.
            If None, no masking is applied.
        return_encoder_states : bool, default=False
            Whether to return the encoder's final hidden states.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, HIDDEN_STATE]
            If return_encoder_states=False:
                Output tensor of shape (batch_size, future_seq_len, output_dim)
            If return_encoder_states=True:
                Tuple of (output, encoder_states) where encoder_states is (h_n, c_n)
        """
        # 1. Encode past sequence
        _, encoder_states = self.encoder(past_sequence)

        # 2. Decode future sequence using encoder context
        decoder_output, _ = self.decoder(future_sequence, encoder_states)

        # 3. Apply self-attention with optional masking
        attention_mask = None
        if decoder_lengths is not None:
            max_length = future_sequence.size(1)

            # Validate decoder lengths
            if (decoder_lengths <= 0).any():
                msg = "All decoder lengths must be positive"
                raise ValueError(msg)
            if (decoder_lengths > max_length).any():
                msg = f"Decoder lengths cannot exceed max_length {max_length}"
                raise IndexError(msg)

            attention_mask = self._create_attention_mask(decoder_lengths, max_length)

        attention_output = self.attention(
            decoder_output, decoder_output, decoder_output, mask=attention_mask
        )

        # 4. Skip connection (gated or simple residual)
        if self.use_gating:
            assert self.gate_add_norm is not None, "GateAddNorm should be initialized"
            combined = self.gate_add_norm(attention_output, decoder_output)
        else:
            assert self.layer_norm is not None, "LayerNorm should be initialized"
            combined = self.layer_norm(decoder_output + attention_output)

        # 5. Final output projection
        output = self.output_head(combined)

        if return_encoder_states:
            return output, encoder_states
        return output
