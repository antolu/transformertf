from __future__ import annotations

import typing

import torch

from ...nn import GateAddNorm, InterpretableMultiHeadAttention
from ...utils.sequence import (
    pack_decoder_sequences,
    pack_encoder_sequences,
    should_use_packing,
    unpack_to_fixed_length,
)
from .._base_transformer import get_attention_mask

__all__ = ["AttentionLSTMModel"]

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class AttentionLSTMModel(torch.nn.Module):
    """
    Attention-enhanced LSTM for sequence-to-sequence time series forecasting.

    This model combines encoder-decoder LSTM architecture with cross-attention mechanism
    where decoder outputs attend to both encoder and decoder sequences. The model includes
    optional gating for skip connections and supports dynamic sequence lengths for
    variable-length samples in batches.

    The architecture consists of:
    1. Encoder LSTM: Processes past feature sequences
    2. Decoder LSTM: Generates future predictions using encoder context
    3. Cross-Attention: Decoder outputs attend to concatenated encoder-decoder sequences
    4. Skip Connection: Optional gating or simple residual connection
    5. Linear Output: Single linear layer for final predictions

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
    d_model : int, default=128
        Hidden size used for both encoder and decoder LSTM layers when d_encoder
        and d_decoder are not specified.
    d_encoder : int, optional
        Hidden size for encoder LSTM layers. Takes precedence over d_model if provided.
    d_decoder : int, optional
        Hidden size for decoder LSTM layers. Takes precedence over d_model if provided.
    num_layers : int, default=2
        Number of LSTM layers used for both encoder and decoder.
    dropout : float, default=0.1
        Dropout probability applied to all components (LSTM, attention, output).
    num_heads : int, default=4
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
        Cross-attention mechanism for decoder outputs.
    state_projection : torch.nn.ModuleDict or None
        Projects encoder states to decoder dimensions (if d_encoder != d_decoder).
    encoder_output_projection : torch.nn.Linear or None
        Projects encoder outputs to decoder dimensions (if d_encoder != d_decoder).
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
    >>> # Create model with shared dimensions
    >>> model = AttentionLSTMModel(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     d_model=64,
    ...     num_layers=2,
    ...     num_heads=4,
    ...     use_gating=True
    ... )
    >>>
    >>> # Create model with different encoder/decoder dimensions
    >>> model = AttentionLSTMModel(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     d_encoder=128,
    ...     d_decoder=64,
    ...     num_layers=2,
    ...     num_heads=4
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
    **Flexible Architecture:**

    Encoder and decoder can use different hidden dimensions (d_encoder, d_decoder)
    for optimal capacity allocation. If not specified, both use d_model. When
    dimensions differ, projection layers handle the dimensional compatibility.

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
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        use_gating: bool = True,
        trainable_add: bool = False,
        output_dim: int = 1,
        causal_attention: bool = True,
        *,
        d_encoder: int | None = None,
        d_decoder: int | None = None,
    ):
        super().__init__()

        # Determine actual encoder and decoder dimensions
        self.d_encoder = d_encoder if d_encoder is not None else d_model
        self.d_decoder = d_decoder if d_decoder is not None else d_model

        # Validate that decoder dimension is divisible by num_heads for attention mechanism
        assert self.d_decoder % num_heads == 0, (
            f"d_decoder ({self.d_decoder}) must be divisible by num_heads ({num_heads}) "
            f"for the attention mechanism to work properly."
        )

        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_gating = use_gating
        self.output_dim = output_dim
        self.trainable_add = trainable_add
        self.causal_attention = causal_attention

        # Encoder LSTM
        self.encoder = torch.nn.LSTM(
            input_size=num_past_features,
            hidden_size=self.d_encoder,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Decoder LSTM
        self.decoder = torch.nn.LSTM(
            input_size=num_future_features,
            hidden_size=self.d_decoder,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # State projection layer (if encoder and decoder have different hidden sizes)
        self.state_projection: torch.nn.Module | None = None
        self.encoder_output_projection: torch.nn.Module | None = None
        if self.d_encoder != self.d_decoder:
            self.state_projection = torch.nn.ModuleDict({
                "h_proj": torch.nn.Linear(self.d_encoder, self.d_decoder),
                "c_proj": torch.nn.Linear(self.d_encoder, self.d_decoder),
            })
            # Project encoder output to decoder dimension for cross-attention
            self.encoder_output_projection = torch.nn.Linear(
                self.d_encoder, self.d_decoder
            )

        # Cross-attention mechanism (operates on decoder dimension)
        self.attention = InterpretableMultiHeadAttention(
            d_model=self.d_decoder,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Skip connection mechanism (operates on decoder dimension)
        if use_gating:
            self.gate_add_norm = GateAddNorm(
                input_dim=self.d_decoder,
                dropout=dropout,
                trainable_add=trainable_add,
            )
            self.layer_norm = None
        else:
            self.gate_add_norm = None
            self.layer_norm = torch.nn.LayerNorm(self.d_decoder)

        # Output projection
        self.output_head = torch.nn.Linear(self.d_decoder, output_dim)

    def _project_encoder_states(self, encoder_states: HIDDEN_STATE) -> HIDDEN_STATE:
        """Project encoder states to decoder dimensions if needed."""
        if self.state_projection is None:
            return encoder_states

        h_enc, c_enc = encoder_states
        h_proj = self.state_projection["h_proj"](h_enc)  # type: ignore[index]
        c_proj = self.state_projection["c_proj"](c_enc)  # type: ignore[index]
        return h_proj, c_proj

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
            Used for attention masking and sequence packing when provided.
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
        use_encoder_packing = should_use_packing(encoder_lengths)
        use_decoder_packing = should_use_packing(decoder_lengths)

        if use_encoder_packing and encoder_lengths is not None:
            packed_encoder_input = pack_encoder_sequences(
                past_sequence,
                encoder_lengths,
                align_first=False,
            )
            packed_encoder_output, encoder_states = self.encoder(packed_encoder_input)
            encoder_output, _ = unpack_to_fixed_length(
                packed_encoder_output, total_length=past_sequence.size(1)
            )
        else:
            encoder_output, encoder_states = self.encoder(past_sequence)

        # Project encoder states to decoder dimensions if needed
        decoder_init_states = self._project_encoder_states(encoder_states)

        if use_decoder_packing and decoder_lengths is not None:
            packed_decoder_input = pack_decoder_sequences(
                future_sequence, decoder_lengths
            )
            packed_decoder_output, _ = self.decoder(
                packed_decoder_input, decoder_init_states
            )
            decoder_output, _ = unpack_to_fixed_length(
                packed_decoder_output, total_length=future_sequence.size(1)
            )
        else:
            decoder_output, _ = self.decoder(future_sequence, decoder_init_states)

        attention_mask = None
        if decoder_lengths is not None and encoder_lengths is not None:
            max_encoder_length = past_sequence.size(1)
            max_decoder_length = future_sequence.size(1)

            # Validate decoder lengths
            if (decoder_lengths <= 0).any():
                msg = "All decoder lengths must be positive"
                raise ValueError(msg)
            if (decoder_lengths > max_decoder_length).any():
                msg = f"Decoder lengths cannot exceed max_length {max_decoder_length}"
                raise IndexError(msg)

            if (encoder_lengths <= 0).any():
                msg = "All encoder lengths must be positive"
                raise ValueError(msg)
            if (encoder_lengths > max_encoder_length).any():
                msg = f"Encoder lengths cannot exceed past_sequence length {max_encoder_length}"
                raise IndexError(msg)

            attention_mask = get_attention_mask(
                encoder_lengths=encoder_lengths,
                decoder_lengths=decoder_lengths,
                max_encoder_length=max_encoder_length,
                max_decoder_length=max_decoder_length,
                causal_attention=self.causal_attention,
            )

        # Project encoder output to decoder dimension if needed
        if self.encoder_output_projection is not None:
            projected_encoder_output = self.encoder_output_projection(encoder_output)
        else:
            projected_encoder_output = encoder_output

        attn_input = torch.concatenate(
            (projected_encoder_output, decoder_output), dim=-2
        )

        attention_output = self.attention(
            decoder_output,
            attn_input,
            attn_input,
            mask=attention_mask,
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
