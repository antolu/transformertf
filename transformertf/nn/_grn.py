"""
Implementation of Gated Residual Network (GRN) module for temporal fusion transformers.

This module implements the Gated Residual Network as described in the Temporal Fusion
Transformer paper by Lim et al. The GRN is a versatile building block that combines
feedforward transformations with gating mechanisms and residual connections to enable
effective feature processing and context integration.

Classes
-------
GatedResidualNetwork : torch.nn.Module
    Context-aware gated residual network with flexible dimension projection.

Notes
-----
The GRN is designed to be used throughout the Temporal Fusion Transformer
architecture for various purposes including static covariate encoding,
variable selection, and temporal processing. Its flexibility in handling
different input/output dimensions and optional context makes it suitable
for diverse architectural components.

The gating mechanism allows the network to selectively control information
flow, while the residual connection ensures stable gradient flow during
training. The optional context input enables the network to condition its
transformations on external information.

References
----------
.. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
   multi-horizon time series forecasting." International Conference on
   Machine Learning. PMLR, 2021. https://arxiv.org/abs/1912.09363
"""

from __future__ import annotations

import typing

import torch

from ._get_activation import VALID_ACTIVATIONS, get_activation
from ._glu import GatedLinearUnit
from ._resample_norm import ResampleNorm


class GatedResidualNetwork(torch.nn.Module):
    """
    Context-aware gated residual network with flexible dimension projection.

    This module implements the Gated Residual Network (GRN) as described in the
    Temporal Fusion Transformer paper. The GRN provides a flexible building block
    that combines feedforward transformations with gating mechanisms and residual
    connections, optionally incorporating context information.

    The mathematical formulation is:

    .. math::
        \\text{GRN}(x, c) = \\text{LayerNorm}(\\text{Proj}(x) + \\text{GLU}(\\phi(\\text{fc}_1(x) + \\text{fc}_3(c))))

    where:
    - :math:`\\text{Proj}(x)` is the projection of input to output dimension
    - :math:`\\phi` is the activation function (default: ELU)
    - :math:`\\text{fc}_1` and :math:`\\text{fc}_3` are linear layers
    - :math:`\\text{GLU}` is the gated linear unit for selective information flow
    - :math:`c` is optional context information

    The detailed computation sequence is:
    1. Project input to match output dimension (if needed)
    2. Apply first linear transformation: :math:`h_1 = \\text{fc}_1(x)`
    3. Add context if provided: :math:`h_2 = h_1 + \\text{fc}_3(c)`
    4. Apply activation: :math:`h_3 = \\phi(h_2)`
    5. Apply second linear transformation: :math:`h_4 = \\text{fc}_2(h_3)`
    6. Apply dropout and GLU: :math:`h_5 = \\text{GLU}(\\text{Dropout}(h_4))`
    7. Add residual and normalize: :math:`\\text{LayerNorm}(\\text{Proj}(x) + h_5)`

    Parameters
    ----------
    input_dim : int
        Input feature dimension. Must match the last dimension of input tensors.
    output_dim : int
        Output feature dimension. The output will have this dimension.
    hidden_dim : int, optional
        Hidden layer dimension for internal transformations. If None, defaults
        to input_dim. Controls the capacity of the internal feedforward network.
    context_dim : int, optional
        Context feature dimension. If provided, enables context-aware processing
        where external information can influence the transformation. The context
        tensor must have this dimension.
    dropout : float, default=0.1
        Dropout probability applied before the GLU. Must be in [0, 1).
        Helps prevent overfitting in the gated transformation.
    activation : VALID_ACTIVATIONS, default="elu"
        Activation function applied to the hidden representation. Common choices
        include "elu", "relu", "gelu", "swish".
    projection : {"linear", "interpolate"}, default="linear"
        Method for projecting input to output dimension when they differ.
        - "linear": Use a linear layer for projection
        - "interpolate": Use temporal interpolation for projection

    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    context_dim : int or None
        Context feature dimension if context is used.
    hidden_dim : int
        Hidden layer dimension.
    fc1 : torch.nn.Linear
        First linear transformation layer.
    fc2 : torch.nn.Linear
        Second linear transformation layer.
    fc3 : torch.nn.Linear or None
        Context linear transformation layer (if context_dim is specified).
    dropout : torch.nn.Dropout
        Dropout layer for regularization.
    project : torch.nn.Module
        Projection layer for dimension matching.
    glu1 : GatedLinearUnit
        Gated linear unit for selective information flow.
    norm : torch.nn.LayerNorm
        Layer normalization for output stabilization.
    activation : callable
        Activation function.

    Notes
    -----
    The GRN is designed to be a versatile building block that can handle:
    - Dimension transformation (input_dim → output_dim)
    - Context integration (optional external conditioning)
    - Gated information flow (selective feature processing)
    - Residual connections (stable gradient flow)

    The context mechanism allows the GRN to condition its transformations on
    external information, making it particularly useful for time series tasks
    where static covariates or temporal context need to influence processing.

    The projection method affects how input dimensions are mapped to output
    dimensions when they differ. Linear projection is more common and
    computationally efficient, while interpolation projection can be useful
    for temporal data where dimension changes represent time axis transformations.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import GatedResidualNetwork
    >>>
    >>> # Basic GRN without context
    >>> grn = GatedResidualNetwork(input_dim=256, output_dim=512)
    >>> x = torch.randn(32, 128, 256)  # (batch, seq, features)
    >>> output = grn(x)
    >>> print(output.shape)  # torch.Size([32, 128, 512])
    >>>
    >>> # GRN with context conditioning
    >>> grn_ctx = GatedResidualNetwork(
    ...     input_dim=256, output_dim=256, context_dim=64
    ... )
    >>> x = torch.randn(32, 128, 256)
    >>> context = torch.randn(32, 128, 64)  # External context
    >>> output_ctx = grn_ctx(x, context)
    >>> print(output_ctx.shape)  # torch.Size([32, 128, 256])
    >>>
    >>> # GRN with custom configuration
    >>> grn_custom = GatedResidualNetwork(
    ...     input_dim=256,
    ...     output_dim=512,
    ...     hidden_dim=1024,
    ...     dropout=0.2,
    ...     activation="gelu",
    ...     projection="interpolate"
    ... )
    >>> output_custom = grn_custom(x)
    >>> print(output_custom.shape)  # torch.Size([32, 128, 512])
    >>>
    >>> # Typical usage in TFT architecture
    >>> # For static covariate encoding
    >>> static_encoder = GatedResidualNetwork(
    ...     input_dim=10,    # Number of static features
    ...     output_dim=256   # Hidden dimension
    ... )
    >>> static_features = torch.randn(32, 10)
    >>> static_encoded = static_encoder(static_features)

    See Also
    --------
    transformertf.nn.GatedLinearUnit : Underlying gated linear unit
    transformertf.nn.VariableSelection : Uses GRN for feature selection
    transformertf.nn.ResampleNorm : Alternative projection method

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
        output_dim: int,
        hidden_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "elu",
        projection: typing.Literal["linear", "interpolate"] = "linear",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        hidden_dim = hidden_dim or input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        self.fc3 = (
            torch.nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim
            else None
        )

        if input_dim != output_dim:
            if projection == "linear":
                self.project = torch.nn.Linear(input_dim, output_dim)
            elif projection == "interpolate":
                self.resample = ResampleNorm(input_dim, output_dim)
            else:
                msg = (
                    f"Unknown resampling method: {projection} "
                    f"(choose from 'linear' or 'interpolate')"
                )
                raise ValueError(msg)
        else:
            self.project = torch.nn.Identity()

        self.glu1 = GatedLinearUnit(output_dim, dropout=dropout)
        self.norm = torch.nn.LayerNorm(output_dim)

        self.activation = get_activation(activation)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        Initialize GRN parameters using appropriate initialization strategies.

        Applies specialized initialization schemes to different parameter types:
        - Bias parameters: Zero initialization
        - fc1/fc2 weights: Kaiming normal initialization for ReLU-like activations
        - fc3 weights: Xavier uniform initialization for context integration

        Notes
        -----
        The initialization strategy is designed to:
        1. Prevent vanishing/exploding gradients during training
        2. Account for different activation functions in the network
        3. Ensure proper signal flow through residual connections

        Kaiming initialization is used for fc1/fc2 as they are followed by
        activations like ELU/ReLU, while Xavier initialization is used for
        fc3 as it handles context integration without immediate activation.

        This method is called automatically during construction but can be
        called manually to reinitialize parameters if needed.

        Examples
        --------
        >>> grn = GatedResidualNetwork(256, 512)
        >>> grn.initialize_parameters()  # Manual reinitialization
        """
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif "fc3" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply gated residual network transformation to input tensor.

        Performs the complete GRN transformation including optional context
        integration, gated processing, and residual connections with
        layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).
            The primary input to be transformed through the GRN.
        context : torch.Tensor, optional
            Context tensor of shape (..., context_dim).
            Optional conditioning information that influences the transformation.
            Required if context_dim was specified during initialization.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim).
            The result of the gated residual transformation with the same
            shape as input except for the last dimension.

        Raises
        ------
        RuntimeError
            If context is None but context_dim was specified during initialization.
        RuntimeError
            If context tensor shape doesn't match expected context_dim.
        RuntimeError
            If input tensor's last dimension doesn't match input_dim.

        Notes
        -----
        The forward pass implements the following computation sequence:
        1. Create residual connection (with projection if needed)
        2. Apply first linear transformation to input
        3. Add context transformation if context is provided
        4. Apply activation function
        5. Apply second linear transformation
        6. Apply dropout for regularization
        7. Apply gated linear unit for selective information flow
        8. Add residual connection and apply layer normalization

        The context integration allows external information to influence
        the hidden representation, enabling conditional processing based
        on static covariates or other contextual factors.

        Examples
        --------
        >>> import torch
        >>> grn = GatedResidualNetwork(256, 512, context_dim=64)
        >>> x = torch.randn(32, 128, 256)
        >>> context = torch.randn(32, 128, 64)
        >>> output = grn(x, context)
        >>> print(output.shape)  # torch.Size([32, 128, 512])
        >>>
        >>> # Without context
        >>> grn_no_ctx = GatedResidualNetwork(256, 512)
        >>> output_no_ctx = grn_no_ctx(x)
        >>> print(output_no_ctx.shape)  # torch.Size([32, 128, 512])
        >>>
        >>> # Check normalization properties
        >>> print(f"Mean: {output.mean(-1).abs().mean():.4f}")  # ≈ 0
        >>> print(f"Std:  {output.std(-1).mean():.4f}")         # ≈ 1
        """
        if hasattr(self, "resample"):
            residual = self.resample(x)
        elif hasattr(self, "project"):
            residual = self.project(x)
        else:
            residual = x

        x = self.fc1(x)
        if self.fc3 is not None:
            x += self.fc3(context)

        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x) if self.dropout is not None else x
        x = self.glu1(x)

        return self.norm(x + residual)
