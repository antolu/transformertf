from __future__ import annotations

import warnings

import torch

from ._get_activation import VALID_ACTIVATIONS
from ._temporal_conv_block import TemporalConvBlock

__all__ = ["TemporalEncoder"]


class TemporalEncoder(torch.nn.Module):
    """
    Temporal encoder for sequence compression using stacked temporal convolution blocks.

    This encoder progressively compresses input sequences using multiple temporal
    convolution blocks with increasing dilation rates and optional downsampling.
    It's designed to reduce sequence length while preserving important temporal
    patterns for efficient attention processing.

    Parameters
    ----------
    input_dim : int
        Number of input features/channels
    d_hidden : int, default=256
        Hidden dimension for convolution blocks
    num_layers : int, default=4
        Number of temporal convolution layers
    kernel_size : int, default=3
        Kernel size for convolutions
    dropout : float, default=0.1
        Dropout probability
    activation : VALID_ACTIVATIONS, default="relu"
        Activation function to use
    compression_factor : int, default=4
        Factor by which to compress the sequence length.
        Final length will be approximately input_length // compression_factor
    max_dilation : int, default=8
        Maximum dilation factor for temporal convolutions

    Notes
    -----
    The encoder architecture:
    1. Projects input to hidden dimension
    2. Applies multiple temporal convolution blocks with increasing dilation
    3. Optional adaptive pooling for exact compression factor
    4. Returns compressed sequence for attention processing

    **IMPORTANT WARNING**: This encoder performs sequence compression within the model.
    Ensure your input sequences are sufficiently long to avoid information loss.
    Minimum recommended input length is compression_factor * max_dilation * kernel_size.

    For typical parameters (compression_factor=4, max_dilation=8, kernel_size=3),
    minimum input length should be at least 96 time steps.

    Examples
    --------
    >>> # Basic encoder for sequence compression
    >>> encoder = TemporalEncoder(input_dim=64, d_hidden=128)
    >>> x = torch.randn(32, 200, 64)  # [batch, seq_len, features]
    >>> compressed = encoder(x)  # [32, 50, 128] (4x compression)

    >>> # Custom compression settings
    >>> encoder = TemporalEncoder(
    ...     input_dim=32,
    ...     d_hidden=256,
    ...     compression_factor=8,
    ...     num_layers=6
    ... )
    >>> x = torch.randn(16, 400, 32)
    >>> compressed = encoder(x)  # [16, 50, 256] (8x compression)
    """

    def __init__(
        self,
        input_dim: int,
        d_hidden: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        compression_factor: int = 4,
        max_dilation: int = 8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.compression_factor = compression_factor
        self.max_dilation = max_dilation

        # Input projection
        self.input_proj = torch.nn.Linear(input_dim, d_hidden)

        # Temporal convolution layers with increasing dilation
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            # Calculate dilation: exponentially increasing up to max_dilation
            dilation = min(2**i, max_dilation)

            self.conv_layers.append(
                TemporalConvBlock(
                    in_channels=d_hidden,
                    out_channels=d_hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                    use_depthwise=True,
                )
            )

        # Compression layer - adaptive pooling for exact compression
        if compression_factor > 1:
            self.compression = torch.nn.AdaptiveAvgPool1d(
                None
            )  # Will be set dynamically
        else:
            self.compression = None

        # Output normalization
        self.output_norm = torch.nn.LayerNorm(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, input_dim]

        Returns
        -------
        torch.Tensor
            Compressed tensor of shape [batch_size, compressed_len, hidden_dim]
            where compressed_len ≈ seq_len // compression_factor

        Raises
        ------
        RuntimeWarning
            If input sequence is too short for effective compression
        """
        _batch_size, seq_len, _ = x.shape

        # Check minimum sequence length
        min_required_len = (
            self.compression_factor * self.max_dilation * 3
        )  # Conservative estimate
        if seq_len < min_required_len:
            warnings.warn(
                f"Input sequence length ({seq_len}) is shorter than recommended minimum "
                f"({min_required_len}) for compression_factor={self.compression_factor} "
                f"and max_dilation={self.max_dilation}. This may result in information loss.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply temporal convolution layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Apply compression if specified
        if self.compression is not None and self.compression_factor > 1:
            target_len = max(1, seq_len // self.compression_factor)

            # Convert to [batch, channels, seq_len] for pooling
            x = x.transpose(1, 2)
            x = torch.nn.functional.adaptive_avg_pool1d(x, target_len)
            # Convert back to [batch, seq_len, channels]
            x = x.transpose(1, 2)

        # Final normalization
        return self.output_norm(x)
