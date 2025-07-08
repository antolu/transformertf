"""
Implementation of GateAddNorm block for gated feedforward networks.

This module implements a compound layer that combines a Gated Linear Unit (GLU)
with residual connections and layer normalization. This pattern is commonly used
in transformer architectures to create expressive feedforward networks with
learnable gating mechanisms.

Classes
-------
GateAddNorm : torch.nn.Module
    Gated feedforward layer with residual connection and layer normalization.

Notes
-----
The GateAddNorm block is designed to replace traditional feedforward layers
in transformer architectures, providing enhanced expressiveness through gating
while maintaining training stability through residual connections and
normalization.

This implementation follows the pattern used in models like the Temporal Fusion
Transformer, where gated feedforward networks are used extensively throughout
the architecture.

References
----------
.. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
   multi-horizon time series forecasting." ICML 2021.
.. [2] Dauphin, Yann N., et al. "Language modeling with gated convolutional
   networks." ICML 2017.
"""

from __future__ import annotations

import torch

from ._add_norm import AddNorm
from ._glu import GatedLinearUnit


class GateAddNorm(torch.nn.Module):
    """
    Gated feedforward layer with residual connection and layer normalization.

    This layer combines a Gated Linear Unit (GLU) with the Add & Norm operation,
    creating a powerful building block for transformer architectures. The layer
    applies gated linear transformations followed by residual connections and
    layer normalization.

    The mathematical operation is:

    .. math::
        \\text{GateAddNorm}(x, r) = \\text{AddNorm}(\\text{GLU}(x), r)

    where:

    .. math::
        \\text{GLU}(x) = (xW_1 + b_1) \\odot \\sigma(xW_2 + b_2)
        \\text{AddNorm}(y, r) = \\text{LayerNorm}(y + r)

    :math:`W_1, W_2` are learned weight matrices, :math:`b_1, b_2` are bias vectors,
    :math:`\\odot` is element-wise multiplication, and :math:`\\sigma` is the sigmoid
    activation function.

    Parameters
    ----------
    input_dim : int
        Input feature dimension. Must match the last dimension of input tensors.
    output_dim : int, optional
        Output feature dimension. If None, defaults to input_dim. The residual
        connection tensor must have this dimension.
    dropout : float, default=0.0
        Dropout probability applied within the GLU. Must be in [0, 1).
    trainable_add : bool, default=False
        Whether to use learnable gating for the residual connection in AddNorm.
        When True, applies sigmoid-gated scaling to the skip connection.

    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    glu : GatedLinearUnit
        Gated linear unit for feature transformation.
    add_norm : AddNorm
        Add & norm layer for residual connection and normalization.

    Notes
    -----
    This layer is particularly effective in transformer architectures where
    it serves as an enhanced replacement for traditional feedforward layers.
    The gating mechanism allows the model to selectively pass information,
    potentially improving representational capacity.

    The combination of GLU and residual connections helps mitigate the vanishing
    gradient problem while providing non-linear transformations that can capture
    complex patterns in the data.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import GateAddNorm
    >>>
    >>> # Basic usage with same input/output dimensions
    >>> gate_add_norm = GateAddNorm(256)
    >>> x = torch.randn(32, 128, 256)  # (batch, seq, features)
    >>> residual = torch.randn(32, 128, 256)
    >>> output = gate_add_norm(x, residual)
    >>> print(output.shape)  # torch.Size([32, 128, 256])
    >>>
    >>> # With dimension change
    >>> gate_add_norm = GateAddNorm(256, output_dim=512)
    >>> x = torch.randn(32, 128, 256)
    >>> residual = torch.randn(32, 128, 512)  # Must match output_dim
    >>> output = gate_add_norm(x, residual)
    >>> print(output.shape)  # torch.Size([32, 128, 512])
    >>>
    >>> # With dropout and trainable residual gating
    >>> gate_add_norm = GateAddNorm(256, dropout=0.1, trainable_add=True)
    >>> output = gate_add_norm(x, residual)
    >>>
    >>> # Typical usage in transformer feedforward block
    >>> # After multi-head attention
    >>> attention_output = torch.randn(32, 128, 256)
    >>> ffn_input = torch.randn(32, 128, 256)  # residual from attention
    >>> ffn_output = gate_add_norm(attention_output, ffn_input)

    See Also
    --------
    transformertf.nn.GatedLinearUnit : Underlying gated linear unit
    transformertf.nn.AddNorm : Underlying add & norm operation
    transformertf.nn.GatedResidualNetwork : Alternative gated architecture

    References
    ----------
    .. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
       multi-horizon time series forecasting." ICML 2021.
    .. [2] Dauphin, Yann N., et al. "Language modeling with gated convolutional
       networks." ICML 2017.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        *,
        trainable_add: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        output_dim = output_dim or input_dim
        self.output_dim = output_dim
        self.glu = GatedLinearUnit(input_dim, output_dim, dropout)

        self.add_norm = AddNorm(output_dim, trainable_add=trainable_add)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Apply gated feedforward transformation with residual connection.

        Applies the GLU transformation to the input tensor, then combines it
        with the residual tensor using the Add & Norm operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).
            The primary input to be transformed through the GLU.
        residual : torch.Tensor
            Residual connection tensor of shape (..., output_dim).
            Must be compatible with the output of the GLU for element-wise addition.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim).
            Result of GLU transformation combined with residual connection
            and layer normalization.

        Raises
        ------
        RuntimeError
            If tensor shapes are incompatible for the GLU transformation
            or residual addition.

        Notes
        -----
        The forward pass performs the following sequence:
        1. Apply GLU to transform input: y = GLU(x)
        2. Add residual and normalize: output = LayerNorm(y + residual)

        This operation is fully differentiable and maintains gradient flow
        through both the transformed path and the residual connection.

        Examples
        --------
        >>> import torch
        >>> gate_add_norm = GateAddNorm(256, output_dim=512)
        >>> x = torch.randn(32, 128, 256)
        >>> residual = torch.randn(32, 128, 512)
        >>> output = gate_add_norm(x, residual)
        >>> print(output.shape)  # torch.Size([32, 128, 512])
        >>>
        >>> # Verify normalization properties
        >>> print(f"Mean: {output.mean(-1).abs().mean():.4f}")  # ≈ 0
        >>> print(f"Std:  {output.std(-1).mean():.4f}")         # ≈ 1
        """
        x = self.glu(x)
        return self.add_norm(x, residual)
