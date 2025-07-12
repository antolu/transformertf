from __future__ import annotations

import torch

from ._get_activation import VALID_ACTIVATIONS, get_activation

__all__ = ["TemporalConvBlock"]


class TemporalConvBlock(torch.nn.Module):
    """
    Temporal convolutional block with dilated convolutions for sequence modeling.

    This block implements the TCNN-inspired temporal convolution approach with
    depthwise separable convolutions, residual connections, and layer normalization.
    It's designed for efficient processing of long sequences in time series models.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, default=3
        Size of the convolutional kernel
    dilation : int, default=1
        Dilation factor for the convolution
    dropout : float, default=0.1
        Dropout probability
    activation : VALID_ACTIVATIONS, default="relu"
        Activation function to use
    use_depthwise : bool, default=True
        Whether to use depthwise separable convolutions for efficiency

    Notes
    -----
    The block follows this architecture:
    1. Optional depthwise convolution (if use_depthwise=True)
    2. Pointwise convolution (1x1 conv)
    3. Layer normalization
    4. Activation function
    5. Dropout
    6. Residual connection (if in_channels == out_channels)

    For long sequences and memory efficiency, depthwise separable convolutions
    are used by default, which significantly reduce the number of parameters
    while maintaining representational capacity.

    **WARNING**: When using this block for sequence compression/decompression,
    ensure your input sequences are long enough to accommodate the dilated
    convolutions. Minimum recommended sequence length is kernel_size * dilation.

    Examples
    --------
    >>> # Basic temporal convolution block
    >>> block = TemporalConvBlock(in_channels=64, out_channels=64)
    >>> x = torch.randn(32, 100, 64)  # [batch, seq_len, channels]
    >>> out = block(x)  # [32, 100, 64]

    >>> # Block with dilation for larger receptive field
    >>> dilated_block = TemporalConvBlock(
    ...     in_channels=64, out_channels=128, dilation=4
    ... )
    >>> out = dilated_block(x)  # [32, 100, 128]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        use_depthwise: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_depthwise = use_depthwise

        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2

        if use_depthwise and in_channels > 1:
            # Depthwise separable convolution: depthwise + pointwise
            self.depthwise = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                groups=in_channels,  # Key for depthwise convolution
                bias=False,
            )
            self.pointwise = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            )
        else:
            # Standard convolution
            self.conv = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=True,
            )

        # Normalization and activation
        self.norm = torch.nn.LayerNorm(out_channels)
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

        # Residual connection (only if dimensions match)
        self.use_residual = in_channels == out_channels
        if not self.use_residual and in_channels != out_channels:
            # Project residual to match output dimensions
            self.residual_proj = torch.nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, in_channels]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, out_channels]
        """
        # Store for residual connection
        residual = x

        # Convert from [batch, seq_len, channels] to [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # Apply convolutions
        if self.use_depthwise and self.in_channels > 1:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)

        # Convert back to [batch, seq_len, channels]
        x = x.transpose(1, 2)

        # Apply normalization, activation, and dropout
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Add residual connection
        if self.use_residual:
            x = x + residual
        elif self.residual_proj is not None:
            x = x + self.residual_proj(residual)

        return x
