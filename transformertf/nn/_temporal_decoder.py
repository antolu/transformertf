from __future__ import annotations

import torch

from ._get_activation import VALID_ACTIVATIONS
from ._temporal_conv_block import TemporalConvBlock

__all__ = ["TemporalDecoder"]


class TemporalDecoder(torch.nn.Module):
    """
    Temporal decoder for sequence reconstruction using deconvolutional layers.

    This decoder reconstructs full-length sequences from compressed representations
    using transposed convolutions and temporal convolution blocks. It's designed
    to expand sequences from attention-processed compressed representations back
    to the target sequence length.

    Parameters
    ----------
    input_dim : int
        Input dimension from compressed representation (typically from attention)
    output_dim : int
        Output dimension for final predictions
    hidden_dim : int, default=256
        Hidden dimension for convolution blocks
    num_layers : int, default=4
        Number of temporal convolution layers after upsampling
    kernel_size : int, default=3
        Kernel size for convolutions
    dropout : float, default=0.1
        Dropout probability
    activation : VALID_ACTIVATIONS, default="relu"
        Activation function to use
    expansion_factor : int, default=4
        Factor by which to expand the sequence length.
        Should match the compression factor used in the encoder

    Notes
    -----
    The decoder architecture:
    1. Projects input to hidden dimension
    2. Applies transposed convolution for upsampling/expansion
    3. Applies multiple temporal convolution blocks for refinement
    4. Projects to output dimension
    5. Adaptive interpolation to exact target length if needed

    The decoder is designed to work with TemporalEncoder, reversing the
    compression process to reconstruct sequences for prediction tasks.

    **WARNING**: The expansion process may introduce artifacts if the
    compressed representation doesn't contain sufficient information.
    Ensure the encoder compression is not too aggressive for your use case.

    Examples
    --------
    >>> # Basic decoder for sequence reconstruction
    >>> decoder = TemporalDecoder(
    ...     input_dim=128,
    ...     output_dim=1
    ... )
    >>> compressed = torch.randn(32, 50, 128)  # Compressed from encoder
    >>> reconstructed = decoder(compressed, tgt_len=200)  # [32, 200, 1]

    >>> # Custom expansion settings
    >>> decoder = TemporalDecoder(
    ...     input_dim=256,
    ...     output_dim=10,
    ...     expansion_factor=8,
    ...     num_layers=6
    ... )
    >>> compressed = torch.randn(16, 50, 256)
    >>> reconstructed = decoder(compressed, tgt_len=400)  # [16, 400, 10]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        expansion_factor: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers

        # Input projection
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        # Upsampling/expansion layer
        if expansion_factor > 1:
            # Transposed convolution for upsampling
            self.upsample = torch.nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=expansion_factor
                * 2,  # Larger kernel for smoother upsampling
                stride=expansion_factor,
                padding=expansion_factor // 2,
                bias=True,
            )
        else:
            self.upsample = None

        # Temporal convolution layers for refinement
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            # Use smaller dilations for refinement (reverse of encoder)
            dilation = max(1, num_layers - i)

            self.conv_layers.append(
                TemporalConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                    use_depthwise=True,
                )
            )

        # Output projection
        self.output_proj = torch.nn.Linear(hidden_dim, output_dim)

        # Final normalization (only if output_dim > 1)
        if output_dim > 1:
            self.output_norm = torch.nn.LayerNorm(output_dim)
        else:
            self.output_norm = None

    def forward(self, x: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """
        Forward pass through the temporal decoder.

        Parameters
        ----------
        x : torch.Tensor
            Compressed input tensor of shape [batch_size, compressed_len, input_dim]
        tgt_len : int
            Target sequence length for the reconstructed output

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape [batch_size, tgt_len, output_dim]
        """
        _batch_size, _compressed_len, _ = x.shape

        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply upsampling if specified
        if self.upsample is not None:
            # Convert to [batch, channels, seq_len] for conv operations
            x = x.transpose(1, 2)
            x = self.upsample(x)
            # Convert back to [batch, seq_len, channels]
            x = x.transpose(1, 2)

        # Apply temporal convolution layers for refinement
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Adjust to exact target length if needed
        current_length = x.shape[1]
        if current_length != tgt_len:
            # Use interpolation to get exact target length
            x = x.transpose(1, 2)  # [batch, channels, seq_len]
            x = torch.nn.functional.interpolate(
                x, size=tgt_len, mode="linear", align_corners=False
            )
            x = x.transpose(1, 2)  # [batch, seq_len, channels]

        # Project to output dimension
        x = self.output_proj(x)

        # Final normalization (only if available)
        if self.output_norm is not None:
            x = self.output_norm(x)

        return x
