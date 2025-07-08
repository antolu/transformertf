"""
Variable Selection Network for interpretable feature selection in transformers.

This module implements a Variable Selection Network (VSN) that provides
interpretable feature selection capabilities for transformer architectures.
The VSN learns to select relevant features dynamically based on input
context, making it particularly useful for time series forecasting with
many potential input features.

Classes
-------
VariableSelection : torch.nn.Module
    Learnable feature selection network with attention-based weighting.

Notes
-----
The Variable Selection Network is a key component of the Temporal Fusion
Transformer that enables interpretable feature selection. It learns to
identify which input features are most relevant for the current prediction
task, providing both improved performance and interpretability.

The network uses gated residual networks to process individual features
and then applies attention-based weighting to select the most relevant
features for the final representation.

References
----------
.. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
   multi-horizon time series forecasting." ICML 2021.
"""

from __future__ import annotations

import torch

from ._grn import GatedResidualNetwork


class VariableSelection(torch.nn.Module):
    """
    Variable Selection Network for interpretable feature selection and processing.

    This module implements a learnable feature selection mechanism that can
    dynamically select the most relevant input features based on context.
    Each feature is processed individually through gated residual networks,
    and then a learned attention mechanism determines the importance weights
    for each feature.

    The mathematical formulation is:

    .. math::
        \\text{VSN}(X, c) = \\sum_{i=1}^{N} w_i \\cdot \\text{GRN}_i(\\text{Pre}_i(X_i))

    where:
    - :math:`X = [X_1, X_2, ..., X_N]` are the input features
    - :math:`\\text{Pre}_i` is the prescaling layer for feature :math:`i`
    - :math:`\\text{GRN}_i` is the gated residual network for feature :math:`i`
    - :math:`w_i` are the learned attention weights: :math:`w = \\text{softmax}(\\text{GRN}_{attn}(\\text{flatten}(\\text{Pre}(X)), c))`
    - :math:`c` is optional context information

    The process consists of:
    1. Prescale each feature individually: :math:`\\tilde{X}_i = \\text{Pre}_i(X_i)`
    2. Process each feature through its own GRN: :math:`H_i = \\text{GRN}_i(\\tilde{X}_i)`
    3. Compute attention weights: :math:`w = \\text{softmax}(\\text{GRN}_{attn}(\\text{flatten}(\\tilde{X}), c))`
    4. Apply weighted sum: :math:`\\text{output} = \\sum_{i=1}^{N} w_i \\cdot H_i`

    Parameters
    ----------
    n_features : int
        Number of input features to select from. Must be positive.
    hidden_dim : int, default=8
        Hidden dimension for prescaling layers. Controls the initial
        feature processing capacity.
    n_dim_model : int, default=300
        Model dimension for output features. This is the final feature
        dimension after selection and processing.
    context_size : int, optional
        Context feature dimension for conditional feature selection.
        If provided, enables context-aware feature selection where
        external information influences feature importance.
    dropout : float, default=0.1
        Dropout probability applied in the gated residual networks.
        Must be in [0, 1).

    Attributes
    ----------
    n_features : int
        Number of input features.
    hidden_dim : int
        Hidden dimension for prescaling.
    n_dim_model : int
        Model output dimension.
    context_size : int or None
        Context dimension if context-aware selection is enabled.
    dropout : torch.nn.Dropout
        Dropout layer.
    prescalers : torch.nn.ModuleList
        List of linear layers for prescaling individual features.
    flattened_grn : GatedResidualNetwork or torch.nn.Identity
        GRN for computing attention weights over features.
    single_grn : torch.nn.ModuleList
        List of GRNs for processing individual features.

    Notes
    -----
    The Variable Selection Network provides several key benefits:
    1. **Interpretability**: Attention weights show which features are important
    2. **Adaptive Selection**: Feature importance can change based on context
    3. **Individual Processing**: Each feature gets its own processing pathway
    4. **Efficient Computation**: Only relevant features contribute to the output

    When n_features=1, the module simplifies to just feature processing
    without selection (unity weights), as there's nothing to select from.

    The prescaling layers transform individual scalar features into higher-
    dimensional representations, allowing the GRNs to learn complex
    transformations for each feature.

    Context-aware selection enables the network to adapt feature importance
    based on external information like static covariates or temporal context.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import VariableSelection
    >>>
    >>> # Basic variable selection
    >>> vs = VariableSelection(n_features=10, n_dim_model=256)\n    >>> x = torch.randn(32, 128, 10)  # (batch, seq, features)
    >>> output, weights = vs(x)
    >>> print(output.shape)  # torch.Size([32, 128, 256])
    >>> print(weights.shape)  # torch.Size([32, 128, 10, 1])
    >>>
    >>> # With context conditioning
    >>> vs_ctx = VariableSelection(
    ...     n_features=10, n_dim_model=256, context_size=64
    ... )
    >>> context = torch.randn(32, 128, 64)
    >>> output_ctx, weights_ctx = vs_ctx(x, context)
    >>> print(output_ctx.shape)  # torch.Size([32, 128, 256])
    >>>
    >>> # Analyze feature importance
    >>> avg_weights = weights.mean(dim=(0, 1))  # Average over batch and time
    >>> feature_importance = avg_weights.squeeze(-1)
    >>> print(f\"Feature importance: {feature_importance}\")
    >>>
    >>> # Single feature case (no selection needed)
    >>> vs_single = VariableSelection(n_features=1, n_dim_model=256)
    >>> x_single = torch.randn(32, 128, 1)
    >>> output_single, weights_single = vs_single(x_single)
    >>> print(weights_single.unique())  # All ones (no selection)
    >>>
    >>> # Time series forecasting usage
    >>> # Select from multiple time series features
    >>> n_series = 20
    >>> vs_forecast = VariableSelection(
    ...     n_features=n_series,
    ...     n_dim_model=512,
    ...     context_size=32  # Static covariates
    ... )
    >>> time_series = torch.randn(64, 168, n_series)  # Weekly data
    >>> static_context = torch.randn(64, 168, 32)
    >>> selected_features, importance = vs_forecast(time_series, static_context)
    >>>
    >>> # Identify most important features
    >>> top_features = importance.mean(dim=(0, 1)).squeeze(-1).argsort(descending=True)[:5]
    >>> print(f\"Top 5 most important features: {top_features}\")

    See Also
    --------
    transformertf.nn.GatedResidualNetwork : Underlying processing networks
    transformertf.nn.InterpretableMultiHeadAttention : Complementary attention mechanism

    References
    ----------
    .. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
       multi-horizon time series forecasting." ICML 2021.
    .. [2] Vaswani, Ashish, et al. "Attention is all you need." NIPS 2017.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 8,
        n_dim_model: int = 300,
        context_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_dim_model = n_dim_model
        self.context_size = context_size
        self.dropout = torch.nn.Dropout(dropout)

        self.prescalers = torch.nn.ModuleList([
            torch.nn.Linear(1, hidden_dim) for _ in range(n_features)
        ])

        if self.n_features > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    input_dim=n_features * hidden_dim,
                    hidden_dim=min(self.n_dim_model, self.n_features),
                    output_dim=self.n_features,
                    context_dim=self.context_size,
                    dropout=dropout,
                    projection="interpolate",
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    input_dim=n_features * hidden_dim,
                    hidden_dim=min(self.n_dim_model, self.n_features),
                    output_dim=self.n_features,
                    dropout=dropout,
                    projection="interpolate",
                )
        else:
            self.flattened_grn = torch.nn.Identity()

        self.single_grn = torch.nn.ModuleList([
            GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=min(hidden_dim, self.n_dim_model),
                output_dim=self.n_dim_model,
                dropout=dropout,
                projection="interpolate",
            )
            for i in range(n_features)
        ])

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply variable selection to input features.

        Performs feature selection by processing each feature individually
        through prescaling and GRN layers, then computing attention weights
        to select the most relevant features for the final output.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor of shape (batch_size, seq_len, n_features).
            Each feature should be a scalar value that will be processed
            through individual prescaling and GRN layers.
        context : torch.Tensor, optional
            Context tensor of shape (batch_size, seq_len, context_size).
            Optional conditioning information that influences feature selection.
            Required if context_size was specified during initialization.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - output : torch.Tensor
                Selected and processed features of shape (batch_size, seq_len, n_dim_model).
                This is the weighted sum of all processed features.
            - weights : torch.Tensor
                Feature importance weights of shape (batch_size, seq_len, n_features, 1).
                These weights sum to 1 across the feature dimension and indicate
                the relative importance of each feature for interpretability.

        Raises
        ------
        ValueError
            If x.shape[-1] != n_features (incorrect number of input features).
        ValueError
            If context is provided but context_size is None, or vice versa.

        Notes
        -----
        The forward pass performs the following steps:
        1. Validate input shapes and context consistency
        2. For each feature:
           a. Apply prescaling: transform scalar to hidden_dim representation
           b. Process through individual GRN: hidden_dim → n_dim_model
        3. If n_features > 1:
           a. Compute attention weights using flattened GRN
           b. Apply softmax to get normalized weights
           c. Apply weighted sum to get final output
        4. If n_features = 1: return processed feature with unity weights

        The attention weights provide interpretability by showing which
        features are most important for the current input context.

        Examples
        --------
        >>> import torch
        >>> vs = VariableSelection(n_features=5, n_dim_model=128)
        >>> x = torch.randn(32, 64, 5)  # batch=32, seq=64, features=5
        >>> output, weights = vs(x)
        >>> print(output.shape)  # torch.Size([32, 64, 128])
        >>> print(weights.shape)  # torch.Size([32, 64, 5, 1])
        >>> print(weights.sum(dim=2).unique())  # All 1.0 (normalized)
        >>>
        >>> # With context
        >>> vs_ctx = VariableSelection(n_features=5, n_dim_model=128, context_size=32)
        >>> context = torch.randn(32, 64, 32)
        >>> output_ctx, weights_ctx = vs_ctx(x, context)
        >>>
        >>> # Analyze feature importance over time
        >>> avg_weights_per_timestep = weights.mean(dim=0)  # Average over batch
        >>> most_important_feature = avg_weights_per_timestep.argmax(dim=1)
        >>> print(f\"Most important feature per timestep: {most_important_feature[:10]}\")
        """
        if x.shape[-1] != self.n_features:
            msg = (
                f"Input tensor must have {self.n_features} features, "
                f"but got {x.shape[-1]} features instead."
            )
            raise ValueError(msg)

        if self.context_size is None and context is not None:
            msg = "Context tensor is not expected, but got a context tensor."
            raise ValueError(msg)

        if self.n_features == 1:
            # if there is only one feature, we don't need to do
            # any variable selection (i.e. unit weights).
            x = self.prescalers[0](x)
            x = self.single_grn[0](x)

            if x.ndim == 3:
                sparse_weights = torch.ones(*x.shape[0:2], 1, 1, device=x.device)
            else:
                sparse_weights = torch.ones(x.shape[0], 1, 1, device=x.device)

            return x, sparse_weights

        # more than one feature
        outputs_l = []
        weights_l = []

        for i in range(self.n_features):
            x_i = self.prescalers[i](x[..., i : i + 1])
            weights_l.append(x_i)

            x_i = self.single_grn[i](x_i)
            outputs_l.append(x_i)

        outputs = torch.stack(outputs_l, dim=-1)
        weights = torch.cat(weights_l, dim=-1)
        weights = self.flattened_grn(weights, context)
        weights = torch.nn.functional.softmax(weights, dim=-1).unsqueeze(-2)

        outputs *= weights
        outputs = outputs.sum(dim=-1)

        return outputs, weights
