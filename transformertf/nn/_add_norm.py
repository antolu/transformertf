"""
Implementation of AddNorm layer for residual connections with layer normalization.

This module implements the Add & Norm operation commonly used in transformer
architectures, combining residual connections with layer normalization. The
implementation includes an optional trainable gating mechanism for the skip
connection.

Classes
-------
AddNorm : torch.nn.Module
    Add & Norm layer with optional trainable skip connection gating.

Notes
-----
The AddNorm operation is a fundamental building block in transformer
architectures, enabling stable training of deep networks through residual
connections combined with layer normalization [1]_.

References
----------
.. [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
   "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).
"""

from __future__ import annotations

import torch


class AddNorm(torch.nn.Module):
    """
    Add & Norm layer implementing residual connections with layer normalization.

    This layer performs the fundamental Add & Norm operation used in transformer
    architectures, combining two tensors through addition followed by layer
    normalization. An optional trainable gating mechanism allows for learnable
    control over the residual connection strength.

    The mathematical formulation depends on the trainable_add parameter:

    - When trainable_add=False (standard):
      .. math::
          \\text{AddNorm}(x, y) = \\text{LayerNorm}(x + y)

    - When trainable_add=True (with gating):
      .. math::
          \\text{AddNorm}(x, y) = \\text{LayerNorm}(x + y \\cdot \\sigma(\\text{mask}) \\cdot 2)

    where :math:`\\sigma` is the sigmoid function and mask is a learnable parameter.

    Parameters
    ----------
    input_size : int
        Dimensionality of input tensors. Both x and y must have this size in their
        last dimension. The output will have the same size.
    trainable_add : bool, default=True
        Whether to use learnable gating for the residual connection. When True,
        applies a sigmoid-gated scaling factor to the skip connection. When False,
        uses standard residual addition.

    Attributes
    ----------
    input_size : int
        The input/output feature dimension.
    trainable_add : bool
        Whether trainable gating is enabled.
    mask : torch.nn.Parameter or None
        Learnable gating parameters when trainable_add=True, None otherwise.
        Shape: (input_size,)
    norm : torch.nn.LayerNorm
        Layer normalization module.

    Notes
    -----
    The trainable gating mechanism scales the sigmoid output by 2 to allow the
    gate to potentially amplify the residual signal beyond the standard [0,1]
    range, providing more flexibility in controlling information flow.

    This implementation is commonly used in transformer architectures where
    it helps stabilize training and enables deeper networks through improved
    gradient flow.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import AddNorm
    >>>
    >>> # Standard Add & Norm
    >>> add_norm = AddNorm(256, trainable_add=False)
    >>> x = torch.randn(32, 128, 256)  # (batch, seq, features)
    >>> y = torch.randn(32, 128, 256)  # residual connection
    >>> output = add_norm(x, y)
    >>> print(output.shape)  # torch.Size([32, 128, 256])
    >>>
    >>> # With trainable gating
    >>> gated_add_norm = AddNorm(256, trainable_add=True)
    >>> output_gated = gated_add_norm(x, y)
    >>> print(output_gated.shape)  # torch.Size([32, 128, 256])
    >>>
    >>> # Typical usage in transformer block
    >>> # After self-attention or feedforward layer
    >>> attention_output = torch.randn(32, 128, 256)
    >>> residual = torch.randn(32, 128, 256)
    >>> normalized = add_norm(attention_output, residual)

    See Also
    --------
    transformertf.nn.GateAddNorm : Combines GLU with Add & Norm
    torch.nn.LayerNorm : Underlying normalization layer

    References
    ----------
    .. [1] He, Kaiming, et al. "Deep residual learning for image recognition."
       CVPR 2016.
    .. [2] Vaswani, Ashish, et al. "Attention is all you need." NIPS 2017.
    """

    def __init__(self, input_size: int, *, trainable_add: bool = True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add

        if self.trainable_add:
            self.mask = torch.nn.Parameter(
                torch.zeros(self.input_size, dtype=torch.float)
            )
        else:
            self.mask = None

        self.norm = torch.nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply Add & Norm operation to input tensors.

        Performs element-wise addition of input tensors followed by layer
        normalization. When trainable_add=True, applies learnable gating
        to the residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Primary input tensor of shape (..., input_size).
            Typically the output of a transformer sub-layer.
        y : torch.Tensor
            Residual/skip connection tensor of shape (..., input_size).
            Must have the same shape as `x`. Typically the input to a
            transformer sub-layer.

        Returns
        -------
        torch.Tensor
            Normalized output tensor of shape (..., input_size).
            Same shape as input tensors.

        Raises
        ------
        RuntimeError
            If x and y have incompatible shapes for addition.

        Notes
        -----
        This operation is differentiable and suitable for gradient-based
        optimization. The layer normalization is applied after the addition,
        which helps stabilize training in deep networks.

        When trainable_add=True, the gating mechanism learns to control
        the contribution of the residual connection, potentially improving
        model expressiveness.

        Examples
        --------
        >>> import torch
        >>> add_norm = AddNorm(256)
        >>> x = torch.randn(2, 10, 256)
        >>> y = torch.randn(2, 10, 256)
        >>> output = add_norm(x, y)
        >>> print(output.shape)  # torch.Size([2, 10, 256])
        >>>
        >>> # Check that layer norm is applied (mean ≈ 0, std ≈ 1)
        >>> print(f"Mean: {output.mean(-1).abs().mean():.4f}")  # ≈ 0
        >>> print(f"Std:  {output.std(-1).mean():.4f}")         # ≈ 1
        """
        if self.trainable_add:
            assert self.mask is not None
            return self.norm(x + y * torch.nn.functional.sigmoid(self.mask) * 2.0)
        return self.norm(x + y)
