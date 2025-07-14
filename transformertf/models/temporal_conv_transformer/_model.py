from __future__ import annotations

import torch

from ...nn import InterpretableMultiHeadAttention, TemporalDecoder, TemporalEncoder
from ...nn._get_activation import VALID_ACTIVATIONS
from .._base_transformer import get_attention_mask

__all__ = ["TemporalConvTransformerModel"]


class TemporalConvTransformerModel(torch.nn.Module):
    """
    Temporal Convolutional Transformer model for efficient long sequence forecasting.

    This model combines temporal convolutions for sequence compression with
    transformer attention for capturing long-range dependencies. It uses an
    encoder-decoder architecture where:
    1. Temporal encoder compresses long input sequences
    2. Attention mechanism processes compressed representations efficiently
    3. Temporal decoder reconstructs full-length predictions

    The model is inspired by TCNN (Temporal Convolutional Neural Networks) and
    designed for efficient processing of very long time series sequences that
    would be computationally prohibitive for standard transformers.

    Parameters
    ----------
    num_past_features : int
        Number of features in encoder (past) sequences
    num_future_features : int
        Number of features in decoder (future) sequences
    output_dim : int, default=1
        Output dimension for predictions
    hidden_dim : int, default=256
        Hidden dimension for convolution and attention layers
    num_attention_heads : int, default=8
        Number of attention heads in the transformer layer
    compression_factor : int, default=4
        Factor by which to compress sequences before attention.
        Higher values = more compression = faster but potentially less accurate
    num_encoder_layers : int, default=4
        Number of temporal convolution layers in encoder
    num_decoder_layers : int, default=4
        Number of temporal convolution layers in decoder
    dropout : float, default=0.1
        Dropout probability
    activation : VALID_ACTIVATIONS, default="relu"
        Activation function to use throughout the model
    max_dilation : int, default=8
        Maximum dilation factor for temporal convolutions

    Attributes
    ----------
    encoder : TemporalEncoder
        Temporal encoder for sequence compression
    decoder : TemporalDecoder
        Temporal decoder for sequence reconstruction
    attention : InterpretableMultiHeadAttention
        Multi-head attention layer for compressed sequences

    Notes
    -----
    **CRITICAL WARNING**: This model performs sequence compression internally,
    which means it requires sufficiently long input sequences to work effectively.

    Minimum recommended sequence lengths:
    - Encoder sequences: >= compression_factor * max_dilation * 12
    - Decoder sequences: >= compression_factor * 8

    For default parameters (compression_factor=4, max_dilation=8):
    - Minimum encoder length: ~384 time steps
    - Minimum decoder length: ~32 time steps

    Using shorter sequences may result in:
    - Information loss during compression
    - Poor model performance
    - Runtime warnings

    Architecture Flow:
    1. Encoder sequences are compressed by temporal encoder
    2. Decoder sequences are compressed by temporal encoder
    3. Both compressed sequences are concatenated for attention
    4. Attention processes the full compressed sequence
    5. Decoder portion is extracted and expanded by temporal decoder
    6. Final predictions are generated

    Examples
    --------
    >>> # Basic model for long sequence forecasting
    >>> model = TemporalConvTransformerModel(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     output_dim=1
    ... )
    >>>
    >>> # Process encoder-decoder batch
    >>> batch = {
    ...     "encoder_input": torch.randn(32, 400, 10),  # Long past sequence
    ...     "decoder_input": torch.randn(32, 100, 5),   # Future sequence
    ...     "encoder_lengths": torch.full((32, 1), 400),
    ...     "decoder_lengths": torch.full((32, 1), 100),
    ... }
    >>> output = model(batch)["output"]  # [32, 100, 1]

    >>> # High compression for very long sequences
    >>> model = TemporalConvTransformerModel(
    ...     num_past_features=20,
    ...     num_future_features=10,
    ...     compression_factor=8,  # More aggressive compression
    ...     hidden_dim=512
    ... )
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        compression_factor: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        max_dilation: int = 8,
        causal_attention: bool = True,
    ):
        super().__init__()

        # Parameter validation
        if compression_factor <= 0:
            msg = f"compression_factor must be positive, got {compression_factor}"
            raise ValueError(msg)

        if hidden_dim % num_attention_heads != 0:
            msg = (
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
            raise ValueError(msg)

        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.compression_factor = compression_factor
        self.causal_attention = causal_attention

        # Temporal encoder for past (encoder) sequences
        self.past_encoder = TemporalEncoder(
            input_dim=num_past_features,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation,
            compression_factor=compression_factor,
            max_dilation=max_dilation,
        )

        # Temporal encoder for future (decoder) sequences
        self.future_encoder = TemporalEncoder(
            input_dim=num_future_features,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation,
            compression_factor=compression_factor,
            max_dilation=max_dilation,
        )

        # Multi-head attention for compressed sequences
        self.attention = InterpretableMultiHeadAttention(
            n_dim_model=hidden_dim,
            n_heads=num_attention_heads,
            dropout=dropout,
        )

        # Temporal decoder for reconstructing predictions
        self.decoder = TemporalDecoder(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            expansion_factor=compression_factor,
        )

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the temporal convolutional transformer.

        Parameters
        ----------
        encoder_input : torch.Tensor
            Encoder input sequences of shape [batch_size, encoder_seq_len, num_past_features]
        decoder_input : torch.Tensor
            Decoder input sequences of shape [batch_size, decoder_seq_len, num_future_features]
        encoder_lengths : torch.Tensor, optional
            Actual lengths of encoder sequences, shape [batch_size, 1]
        decoder_lengths : torch.Tensor, optional
            Actual lengths of decoder sequences, shape [batch_size, 1]

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - output: [batch_size, decoder_seq_len, output_dim]
            - attention_weights: [batch_size, num_heads, compressed_decoder_len, compressed_total_len]

        Raises
        ------
        RuntimeWarning
            If input sequences are too short for effective compression
        """

        batch_size, encoder_seq_len, _ = encoder_input.shape
        _, decoder_seq_len, _ = decoder_input.shape

        # Get sequence lengths (use full length if not provided)
        if encoder_lengths is None:
            encoder_lengths = torch.full(
                (batch_size, 1), encoder_seq_len, device=encoder_input.device
            )
        if decoder_lengths is None:
            decoder_lengths = torch.full(
                (batch_size, 1), decoder_seq_len, device=decoder_input.device
            )

        # Compress both encoder and decoder sequences
        compressed_encoder = self.past_encoder(
            encoder_input
        )  # [batch, compressed_enc_len, hidden_dim]
        compressed_decoder = self.future_encoder(
            decoder_input
        )  # [batch, compressed_dec_len, hidden_dim]

        # Get compressed lengths for attention masking
        compressed_enc_len = compressed_encoder.shape[1]
        compressed_dec_len = compressed_decoder.shape[1]

        # Calculate compressed lengths for masking
        compressed_encoder_lengths = torch.ceil(
            encoder_lengths.float() / self.compression_factor
        ).long()
        compressed_decoder_lengths = torch.ceil(
            decoder_lengths.float() / self.compression_factor
        ).long()

        # Create attention mask for compressed sequences
        attention_mask = get_attention_mask(
            encoder_lengths=compressed_encoder_lengths.squeeze(-1),
            decoder_lengths=compressed_decoder_lengths.squeeze(-1),
            max_encoder_length=compressed_enc_len,
            max_decoder_length=compressed_dec_len,
            causal_attention=self.causal_attention,
        )

        # Concatenate encoder and decoder for attention
        # Shape: [batch, compressed_dec_len, compressed_enc_len + compressed_dec_len]
        combined_sequence = torch.cat([compressed_encoder, compressed_decoder], dim=1)

        # Apply attention - only to decoder portion
        attended_output, attention_weights = self.attention(
            compressed_decoder,
            combined_sequence,
            combined_sequence,
            attention_mask,
            return_attn=True,
        )

        # Reconstruct full-length predictions
        output = self.decoder(
            attended_output, tgt_len=decoder_seq_len
        )  # [batch, decoder_seq_len, output_dim]

        return {
            "output": output,
            "attention_weights": attention_weights,
        }
