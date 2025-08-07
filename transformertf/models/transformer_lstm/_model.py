from __future__ import annotations

import typing

import torch

from ...nn import GateAddNorm, GatedResidualNetwork, InterpretableMultiHeadAttention
from ...utils.sequence import (
    pack_decoder_sequences,
    pack_encoder_sequences,
    should_use_packing,
    unpack_to_fixed_length,
)
from .._base_transformer import get_attention_mask

__all__ = ["TransformerLSTMModel"]

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class TransformerBlock(torch.nn.Module):
    """
    Transformer block with encoder self-attention and cross-attention for TransformerLSTM.

    This block implements a simplified transformer architecture with:
    1. Encoder self-attention using proper masking for variable sequence lengths
    2. Cross-attention between decoder and full sequence (encoder + decoder)
    3. GLU with residual connections for each attention operation

    Parameters
    ----------
    d_model : int
        Model dimension for the model and attention layers.
    num_heads : int
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability for attention and GLU layers.

    Attributes
    ----------
    encoder_self_attention : InterpretableMultiHeadAttention
        Self-attention layer for encoder sequences.
    cross_attention : InterpretableMultiHeadAttention
        Cross-attention layer between decoder and full sequence.
    encoder_gate_add_norm : GateAddNorm
        GateAddNorm layer for encoder self-attention output with residual connection.
    decoder_gate_add_norm : GateAddNorm
        GateAddNorm layer for cross-attention output with residual connection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Validate that d_model is divisible by num_heads
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        # Encoder self-attention
        self.encoder_self_attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention between decoder and full sequence
        self.cross_attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # GateAddNorm layers for residual connections with layer normalization
        self.encoder_gate_add_norm = GateAddNorm(
            input_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        self.decoder_gate_add_norm = GateAddNorm(
            input_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer block.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Encoder output of shape (batch_size, encoder_seq_len, d_model).
        decoder_output : torch.Tensor
            Decoder output of shape (batch_size, decoder_seq_len, d_model).
        attention_mask : torch.Tensor or None
            Attention mask for cross-attention of shape
            (batch_size, decoder_seq_len, encoder_seq_len + decoder_seq_len).
        encoder_lengths : torch.Tensor or None
            Tensor of shape (batch_size,) containing actual encoder sequence lengths
            for creating encoder self-attention masks.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated (encoder_output, decoder_output) after transformer block processing.
        """

        # 1. Encoder self-attention with masking for variable lengths
        # Create encoder self-attention mask using get_attention_mask
        encoder_self_mask = None
        if encoder_lengths is not None:
            encoder_seq_len = encoder_output.size(1)

            # Use get_attention_mask with encoder_lengths twice for encoder self-attention
            encoder_mask_full = get_attention_mask(
                encoder_lengths=encoder_lengths,
                decoder_lengths=encoder_lengths,  # Pass encoder_lengths twice
                max_encoder_length=encoder_seq_len,
                max_decoder_length=encoder_seq_len,  # Same as max_encoder_length
                causal_attention=False,  # No causality for encoder self-attention
            )
            # Extract just the encoder self-attention part
            encoder_self_mask = encoder_mask_full[:, :, :encoder_seq_len]

        # encoder attention + residual connection
        encoder_attn_output = self.encoder_self_attention(
            encoder_output, encoder_output, encoder_output, mask=encoder_self_mask
        )
        encoder_output = self.encoder_gate_add_norm(encoder_attn_output, encoder_output)

        # decoder cross attention + residual connection
        full_sequence = torch.cat([encoder_output, decoder_output], dim=1)
        decoder_attn_output = self.cross_attention(
            decoder_output, full_sequence, full_sequence, mask=attention_mask
        )
        decoder_output = self.decoder_gate_add_norm(decoder_attn_output, decoder_output)

        return encoder_output, decoder_output


class TransformerLSTMModel(torch.nn.Module):
    """
    Transformer-enhanced LSTM for sequence-to-sequence time series forecasting.

    This model combines LSTM encoder-decoder architecture with multiple transformer blocks
    featuring self-attention and cross-attention mechanisms. It processes sequences through
    LSTM layers first, then applies N transformer blocks for enhanced sequence modeling.

    The architecture consists of:
    1. Encoder LSTM: Processes past feature sequences
    2. Decoder LSTM: Generates future predictions using encoder context
    3. N Transformer Blocks: Each with encoder self-attention and cross-attention
    4. Linear Output: Final prediction layer

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
    d_model : int, default=128
        Model dimension used for LSTM layers and transformer blocks.
    num_layers : int, default=2
        Number of LSTM layers for both encoder and decoder.
    num_transformer_blocks : int, default=2
        Number of transformer blocks to apply after LSTM processing.
    dropout : float, default=0.1
        Dropout probability applied to all components.
    num_heads : int, default=4
        Number of attention heads in transformer blocks.
    output_dim : int, default=1
        Output dimension of the model.
    causal_attention : bool, default=True
        Whether to use causal attention masking.

    Attributes
    ----------
    encoder : torch.nn.LSTM
        The encoder LSTM network.
    decoder : torch.nn.LSTM
        The decoder LSTM network.
    transformer_blocks : torch.nn.ModuleList
        List of transformer blocks for sequential processing.
    output_head : torch.nn.Linear
        Linear layer for final output projection.
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        *,
        d_model: int = 128,
        num_layers: int = 2,
        num_transformer_blocks: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        output_dim: int = 1,
        causal_attention: bool = True,
        share_lstm_weights: bool = False,
    ):
        super().__init__()

        # Validate that d_model is divisible by num_heads
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.causal_attention = causal_attention
        self.share_lstm_weights = share_lstm_weights

        self.encoder_grn = GatedResidualNetwork(
            input_dim=num_past_features,
            d_hidden=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        self.decoder_grn = GatedResidualNetwork(
            input_dim=num_future_features,
            d_hidden=d_model,
            output_dim=d_model,
            dropout=dropout,
        )

        if share_lstm_weights:
            # Use the same LSTM for both encoder and decoder
            self.encoder = self.decoder = torch.nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            # Encoder LSTM
            self.encoder = torch.nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )

            # Decoder LSTM
            self.decoder = torch.nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )

        # Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_transformer_blocks)
        ])

        # Output projection
        self.output_head = torch.nn.Linear(d_model, output_dim)

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
        Forward pass through the transformer-enhanced LSTM model.

        Parameters
        ----------
        past_sequence : torch.Tensor
            Past sequence tensor of shape (batch_size, past_seq_len, num_past_features).
        future_sequence : torch.Tensor
            Future sequence tensor of shape (batch_size, future_seq_len, num_future_features).
        encoder_lengths : torch.Tensor or None, default=None
            Tensor of shape (batch_size,) containing actual encoder sequence lengths.
        decoder_lengths : torch.Tensor or None, default=None
            Tensor of shape (batch_size,) containing actual decoder sequence lengths.
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
        # Determine if we should use packed sequences for efficiency
        use_encoder_packing = should_use_packing(encoder_lengths)
        use_decoder_packing = should_use_packing(decoder_lengths)

        # 1. Encode past sequence with optional packing
        encoder_grn_output = self.encoder_grn(past_sequence)

        if use_encoder_packing and encoder_lengths is not None:
            # Pack encoder sequences for efficient LSTM processing
            packed_encoder_input = pack_encoder_sequences(
                encoder_grn_output,
                encoder_lengths,
                align_first=False,  # Already aligned by collate_fn
            )
            packed_encoder_output, encoder_states = self.encoder(packed_encoder_input)
            # Unpack for consistent tensor format
            encoder_output, _ = unpack_to_fixed_length(
                packed_encoder_output, total_length=past_sequence.size(1)
            )
        else:
            # Standard LSTM processing
            encoder_output, encoder_states = self.encoder(encoder_grn_output)

        encoder_output = encoder_output + encoder_grn_output

        # 2. Decode future sequence using encoder context with optional packing
        decoder_grn_output = self.decoder_grn(future_sequence)

        if use_decoder_packing and decoder_lengths is not None:
            # Pack decoder sequences for efficient LSTM processing
            packed_decoder_input = pack_decoder_sequences(
                decoder_grn_output, decoder_lengths
            )
            packed_decoder_output, _ = self.decoder(
                packed_decoder_input, encoder_states
            )
            # Unpack for consistent tensor format
            decoder_output, _ = unpack_to_fixed_length(
                packed_decoder_output, total_length=future_sequence.size(1)
            )
        else:
            # Standard LSTM processing
            decoder_output, _ = self.decoder(decoder_grn_output, encoder_states)

        decoder_output = decoder_output + decoder_grn_output

        # 3. Create attention mask for transformer blocks
        attention_mask = None
        if encoder_lengths is not None and decoder_lengths is not None:
            max_encoder_length = past_sequence.size(1)
            max_decoder_length = future_sequence.size(1)

            # Validate lengths
            if (encoder_lengths <= 0).any():
                msg = "All encoder lengths must be positive"
                raise ValueError(msg)
            if (encoder_lengths > max_encoder_length).any():
                msg = f"Encoder lengths cannot exceed past_sequence length {max_encoder_length}"
                raise IndexError(msg)

            if (decoder_lengths <= 0).any():
                msg = "All decoder lengths must be positive"
                raise ValueError(msg)
            if (decoder_lengths > max_decoder_length).any():
                msg = f"Decoder lengths cannot exceed max_length {max_decoder_length}"
                raise IndexError(msg)

            attention_mask = get_attention_mask(
                encoder_lengths=encoder_lengths,
                decoder_lengths=decoder_lengths,
                max_encoder_length=max_encoder_length,
                max_decoder_length=max_decoder_length,
                causal_attention=self.causal_attention,
            )

        # 4. Apply transformer blocks sequentially
        current_encoder = encoder_output
        current_decoder = decoder_output

        for transformer_block in self.transformer_blocks:
            current_encoder, current_decoder = transformer_block(
                current_encoder, current_decoder, attention_mask, encoder_lengths
            )

        # 5. Final output projection
        output = self.output_head(current_decoder)

        if return_encoder_states:
            return output, encoder_states
        return output
