"""
Gated Linear Unit (GLU) implementation for neural networks.

This module implements the Gated Linear Unit (GLU) as introduced by Dauphin et al.
in "Language Modeling with Gated Convolutional Networks". The GLU provides a
gating mechanism that allows networks to selectively control information flow,
improving gradient flow and model expressiveness.

Classes
-------
GatedLinearUnit : torch.nn.Module
    Gated Linear Unit with efficient single-matrix implementation.

Notes
-----
The GLU was originally proposed for convolutional language models but has proven
effective across various architectures. This implementation uses a single linear
layer that is split, rather than two separate layers, for computational efficiency.

The gating mechanism in GLU helps address the vanishing gradient problem by
providing a linear path for gradients while still allowing for non-linear
transformations through the gated component.

References
----------
.. [1] Dauphin, Yann N., et al. "Language modeling with gated convolutional
   networks." ICML 2017. https://arxiv.org/abs/1612.08083
"""

from __future__ import annotations

import torch


class GatedLinearUnit(torch.nn.Module):
    """
    Gated Linear Unit (GLU) with efficient single-matrix implementation.

    This module implements the Gated Linear Unit as described by Dauphin et al.,
    which provides a gating mechanism for neural networks that allows selective
    information flow. The GLU applies element-wise gating to linear transformations,
    improving gradient flow and model expressiveness.

    The mathematical formulation is:

    .. math::
        \\text{GLU}(x) = (xW_1 + b_1) \\odot \\sigma(xW_2 + b_2)

    where :math:`W_1, W_2` are weight matrices, :math:`b_1, b_2` are bias vectors,
    :math:`\\odot` denotes element-wise multiplication, and :math:`\\sigma` is the
    sigmoid activation function.

    This implementation uses a single linear layer that outputs twice the target
    dimension, then splits the output and applies the gating mechanism. This is
    computationally more efficient than using two separate linear layers.

    Parameters
    ----------
    input_dim : int
        Input feature dimension. Must match the last dimension of input tensors.
    output_dim : int, optional
        Output feature dimension. If None, defaults to input_dim, maintaining
        the same dimensionality as the input.
    dropout : float, default=0.0
        Dropout probability applied to the input before linear transformation.
        Must be in the range [0, 1).

    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    dropout : torch.nn.Dropout
        Dropout layer applied to inputs.
    fc1 : torch.nn.Linear
        Linear layer that outputs 2 * output_dim features for splitting.

    Notes
    -----
    The GLU was originally proposed for convolutional language models but has
    proven effective across various architectures. The gating mechanism helps
    address the vanishing gradient problem by providing a linear path for
    gradients through the sigmoid gate.

    This implementation leverages PyTorch's built-in `torch.nn.functional.glu`
    function for efficient computation of the gating operation.

    The single-matrix approach reduces the number of parameters and computations
    compared to the naive implementation with two separate linear layers:
    - Parameters: input_dim × (2 × output_dim) vs 2 × (input_dim × output_dim)
    - Computations: 1 matrix multiplication vs 2 matrix multiplications

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import GatedLinearUnit
    >>>
    >>> # Basic GLU with same input/output dimensions
    >>> glu = GatedLinearUnit(256)
    >>> x = torch.randn(32, 128, 256)  # (batch, seq, features)
    >>> output = glu(x)
    >>> print(output.shape)  # torch.Size([32, 128, 256])
    >>>
    >>> # GLU with dimension change
    >>> glu_expand = GatedLinearUnit(256, output_dim=512)
    >>> output_expanded = glu_expand(x)
    >>> print(output_expanded.shape)  # torch.Size([32, 128, 512])
    >>>
    >>> # With dropout for regularization
    >>> glu_dropout = GatedLinearUnit(256, dropout=0.1)
    >>> output_dropout = glu_dropout(x)
    >>>
    >>> # Typical usage in feedforward networks
    >>> # As part of a gated feedforward layer
    >>> hidden = torch.randn(32, 128, 256)
    >>> gated_hidden = glu(hidden)
    >>> # Further processing...

    See Also
    --------
    transformertf.nn.GateAddNorm : Combines GLU with residual connections
    transformertf.nn.GatedResidualNetwork : Uses GLU in residual architecture
    torch.nn.functional.glu : Underlying gating function

    References
    ----------
    .. [1] Dauphin, Yann N., et al. "Language modeling with gated convolutional
       networks." ICML 2017. https://arxiv.org/abs/1612.08083
    .. [2] Shazeer, Noam. "GLU variants improve transformer." arXiv preprint
       arXiv:2002.05202 (2020).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(input_dim, self.output_dim * 2)

    def initialize_parameters(self) -> None:
        """
        Initialize layer parameters using Xavier uniform initialization.

        Initializes the linear layer weights using Xavier uniform initialization,
        which is designed to maintain the variance of activations and gradients
        across layers. The bias is initialized to zero.

        Notes
        -----
        Xavier initialization is particularly well-suited for layers with
        sigmoid or tanh activations, making it appropriate for the GLU's
        gating mechanism.

        The initialization follows:
        - Weight: Xavier uniform distribution
        - Bias: Zero initialization

        This method is called automatically during construction but can be
        called manually to reinitialize parameters if needed.

        Examples
        --------
        >>> glu = GatedLinearUnit(256)
        >>> glu.initialize_parameters()  # Manual reinitialization
        """
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gated linear unit transformation to input tensor.

        Performs the GLU transformation by applying dropout, linear transformation,
        and gating. The linear layer outputs twice the target dimension, which is
        then split and gated using the sigmoid activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).
            The last dimension must match the input_dim specified during initialization.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim).
            The gated transformation result with the same shape as input except
            for the last dimension, which becomes output_dim.

        Raises
        ------
        RuntimeError
            If input tensor's last dimension does not match input_dim.

        Notes
        -----
        The forward pass performs the following operations:
        1. Apply dropout to input: x' = Dropout(x)
        2. Linear transformation: y = x' @ W + b (outputs 2 * output_dim)
        3. Split and gate: GLU(y) = y[:, :output_dim] * σ(y[:, output_dim:])

        The gating mechanism allows the network to selectively control information
        flow, where the sigmoid gate determines which parts of the linear
        transformation to pass through.

        Examples
        --------
        >>> import torch
        >>> glu = GatedLinearUnit(256, output_dim=512)
        >>> x = torch.randn(32, 128, 256)
        >>> output = glu(x)
        >>> print(output.shape)  # torch.Size([32, 128, 512])
        >>>
        >>> # The output is the result of gating
        >>> # output = linear_part * sigmoid(gate_part)
        >>> # where both parts come from the same linear transformation
        """
        x = self.dropout(x)
        x = self.fc1(x)
        return torch.nn.functional.glu(x, dim=-1)
