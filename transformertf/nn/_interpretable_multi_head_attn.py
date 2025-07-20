"""
Interpretable Multi-Head Attention implementation for enhanced attention interpretability.

This module implements a specialized multi-head attention mechanism designed
for interpretability in transformer architectures. Unlike standard multi-head
attention, this implementation uses separate query/key projections for each
head while sharing the value projection, making attention patterns more
interpretable.

Classes
-------
InterpretableMultiHeadAttention : torch.nn.Module
    Multi-head attention with enhanced interpretability features.

Notes
-----
The interpretable multi-head attention mechanism is specifically designed
for applications where understanding attention patterns is crucial, such as
in the Temporal Fusion Transformer for time series forecasting. The shared
value projection and separate query/key projections per head allow for
clearer interpretation of what each attention head is focusing on.

This implementation differs from standard multi-head attention in several
key ways:
1. Each head has its own query and key projection layers
2. Value projection is shared across all heads
3. Attention weights are computed and stored for interpretability
4. The final output is the mean of all head outputs

References
----------
.. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
   multi-horizon time series forecasting." ICML 2021.
.. [2] Vaswani, Ashish, et al. "Attention is all you need." NIPS 2017.
"""

from __future__ import annotations

import typing

import torch

from ._scaled_dot_product_attn import ScaledDotProductAttention


class InterpretableMultiHeadAttention(torch.nn.Module):
    """
    Interpretable Multi-Head Attention with enhanced attention visualization capabilities.

    This implementation provides a specialized multi-head attention mechanism
    designed for interpretability in transformer architectures. Unlike standard
    multi-head attention, this module uses separate query/key projections for
    each head while sharing the value projection, making attention patterns
    more interpretable and analyzable.

    The mathematical formulation is:

    .. math::
        \\text{IMHA}(Q, K, V) = \\frac{1}{h} \\sum_{i=1}^h \\text{head}_i W_O

    where for each head :math:`i`:

    .. math::
        \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW^V)

    and the attention mechanism is:

    .. math::
        \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V

    Key differences from standard multi-head attention:
    1. Each head has separate query and key projections: :math:`W_i^Q, W_i^K`
    2. Value projection is shared across all heads: :math:`W^V`
    3. Head outputs are averaged rather than concatenated
    4. Attention weights are preserved for interpretability analysis

    Parameters
    ----------
    n_dim_model : int
        Model dimension (d_model). Must be divisible by n_heads.
        This is the input and output feature dimension.
    n_heads : int
        Number of attention heads. Each head will have dimension d_model // n_heads.
        Must be a positive integer that divides n_dim_model.
    dropout : float, default=0.1
        Dropout probability applied to attention weights and final output.
        Must be in the range [0, 1).

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    d_model : int
        Model dimension.
    dropout : float
        Dropout probability.
    d_q : int
        Query dimension per head (d_model // n_heads).
    d_k : int
        Key dimension per head (d_model // n_heads).
    d_v : int
        Value dimension per head (d_model // n_heads).
    query_layers : torch.nn.ModuleList
        List of linear layers for query projections, one per head.
    key_layers : torch.nn.ModuleList
        List of linear layers for key projections, one per head.
    value_layer : torch.nn.Linear
        Shared linear layer for value projection.
    output_layer : torch.nn.Linear
        Final output projection layer.

    Notes
    -----
    The interpretable design allows for better understanding of attention patterns
    because:
    1. Each head's attention can be analyzed independently
    2. The shared value projection means heads differ only in their attention patterns
    3. Attention weights are preserved and can be visualized
    4. The averaging operation allows for clearer head importance analysis

    This implementation uses scaled dot-product attention with optional masking
    for applications like causal self-attention in time series forecasting.

    The dimension per head is automatically computed as d_model // n_heads,
    which means d_model must be divisible by n_heads for proper operation.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import InterpretableMultiHeadAttention
    >>>
    >>> # Basic usage
    >>> attn = InterpretableMultiHeadAttention(
    ...     n_dim_model=512, n_heads=8, dropout=0.1
    ... )
    >>>
    >>> # Self-attention
    >>> x = torch.randn(32, 128, 512)  # (batch, seq_len, d_model)
    >>> output = attn(x, x, x)
    >>> print(output.shape)  # torch.Size([32, 128, 512])
    >>>
    >>> # Cross-attention with different key/value
    >>> query = torch.randn(32, 64, 512)
    >>> key = torch.randn(32, 128, 512)
    >>> value = torch.randn(32, 128, 512)
    >>> output = attn(query, key, value)
    >>> print(output.shape)  # torch.Size([32, 64, 512])
    >>>
    >>> # With attention weights for interpretability
    >>> output, attn_weights = attn(x, x, x, return_attn=True)
    >>> print(attn_weights.shape)  # torch.Size([32, 128, 8]) for 8 heads
    >>>
    >>> # With causal mask for time series
    >>> seq_len = 128
    >>> mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    >>> output = attn(x, x, x, mask=mask)
    >>>
    >>> # Analyze attention patterns per head
    >>> _, attn_weights = attn(x, x, x, return_attn=True)
    >>> for head_idx in range(8):
    ...     head_attn = attn_weights[:, :, head_idx]
    ...     # Analyze this head's attention pattern
    ...     print(f"Head {head_idx} attention entropy: {head_attn.entropy().mean()}")

    See Also
    --------
    torch.nn.MultiheadAttention : Standard PyTorch multi-head attention
    transformertf.nn.VariableSelection : Uses attention for feature selection

    References
    ----------
    .. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
       multi-horizon time series forecasting." ICML 2021.
    .. [2] Vaswani, Ashish, et al. "Attention is all you need." NIPS 2017.
    .. [3] Wiegreffe, Sarah, and Yuval Pinter. "Attention is not not explanation."
       EMNLP 2019.
    """

    def __init__(
        self,
        n_dim_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = n_heads
        self.d_model = n_dim_model
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

        self.d_q = self.d_k = self.d_v = n_dim_model // n_heads

        self.query_layers = torch.nn.ModuleList([
            torch.nn.Linear(n_dim_model, self.d_q) for _ in range(n_heads)
        ])
        self.key_layers = torch.nn.ModuleList([
            torch.nn.Linear(n_dim_model, self.d_k) for _ in range(n_heads)
        ])
        self.value_layer = torch.nn.Linear(n_dim_model, self.d_v)

        self.output_layer = torch.nn.Linear(self.d_v, n_dim_model)

        # Use custom attention for interpretability (returns attention weights)
        # Dropout is applied at the output level, not within attention
        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        Initialize attention parameters using Xavier uniform initialization.

        Applies Xavier uniform initialization to weight parameters and zero
        initialization to bias parameters. This initialization scheme is
        well-suited for attention mechanisms as it helps maintain stable
        gradients during training.

        Notes
        -----
        Xavier initialization is particularly appropriate for attention layers
        because it:
        1. Maintains the variance of activations across layers
        2. Prevents vanishing/exploding gradients in deep networks
        3. Works well with the softmax operation in attention

        The initialization is applied to:
        - Query projection weights: Xavier uniform
        - Key projection weights: Xavier uniform
        - Value projection weights: Xavier uniform
        - Output projection weights: Xavier uniform
        - All bias parameters: Zero initialization

        This method is called automatically during construction but can be
        called manually to reinitialize parameters if needed.

        Examples
        --------
        >>> attn = InterpretableMultiHeadAttention(512, 8)
        >>> attn.initialize_parameters()  # Manual reinitialization
        """
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.constant_(p, 0)

    @typing.overload
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attn: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        return_attn: typing.Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply interpretable multi-head attention to input tensors.

        Computes attention for each head separately using individual query/key
        projections and a shared value projection. The attention weights are
        preserved for interpretability analysis.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len_q, d_model).
            Contains the queries for attention computation.
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len_k, d_model).
            Contains the keys for attention computation. For self-attention,
            this is the same as query.
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len_v, d_model).
            Contains the values to be aggregated. For self-attention,
            this is the same as query. seq_len_v must equal seq_len_k.
        mask : torch.Tensor, optional
            Attention mask of shape (seq_len_q, seq_len_k) or broadcastable shape.
            If provided, positions with True values will be masked out (ignored).
            Commonly used for causal attention in time series.
        return_attn : bool, default=False
            Whether to return attention weights along with the output.
            When True, returns (output, attention_weights).

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If return_attn=False:
                Output tensor of shape (batch_size, seq_len_q, d_model).
            If return_attn=True:
                Tuple containing:
                - Output tensor of shape (batch_size, seq_len_q, d_model)
                - Attention weights of shape (batch_size, seq_len_q, num_heads)
                  for interpretability analysis

        Raises
        ------
        RuntimeError
            If input tensors have incompatible shapes.
        ValueError
            If d_model is not divisible by num_heads.

        Notes
        -----
        The forward pass performs the following steps:
        1. Apply shared value projection to the value tensor
        2. For each head:
           a. Apply head-specific query and key projections
           b. Compute attention weights using scaled dot-product attention
           c. Apply attention to the projected values
        3. Average the outputs from all heads
        4. Apply final output projection and dropout

        The attention computation uses PyTorch's scaled_dot_product_attention
        for efficiency and to leverage optimized implementations when available.

        The attention weights returned are the raw attention matrices for each
        head, which can be used for visualization and interpretation of what
        each head is attending to.

        Examples
        --------
        >>> import torch
        >>> attn = InterpretableMultiHeadAttention(512, 8)
        >>>
        >>> # Self-attention
        >>> x = torch.randn(32, 128, 512)
        >>> output = attn(x, x, x)
        >>> print(output.shape)  # torch.Size([32, 128, 512])
        >>>
        >>> # With attention weights for analysis
        >>> output, attn_weights = attn(x, x, x, return_attn=True)
        >>> print(attn_weights.shape)  # torch.Size([32, 128, 8])
        >>>
        >>> # Cross-attention
        >>> query = torch.randn(32, 64, 512)
        >>> key_value = torch.randn(32, 128, 512)
        >>> output = attn(query, key_value, key_value)
        >>> print(output.shape)  # torch.Size([32, 64, 512])
        >>>
        >>> # With causal mask
        >>> seq_len = 128
        >>> mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        >>> output = attn(x, x, x, mask=mask)
        >>>
        >>> # Analyze attention patterns
        >>> _, attn_weights = attn(x, x, x, return_attn=True)
        >>> avg_attn = attn_weights.mean(dim=0)  # Average over batch
        >>> head_importance = avg_attn.mean(dim=0)  # Average over sequence
        >>> print(f"Head importance: {head_importance}")
        """
        value = self.value_layer(value)

        heads = []
        attns = []
        for i in range(self.num_heads):
            q = self.query_layers[i](query)
            k = self.key_layers[i](key)

            # Use custom attention implementation that returns both output and weights
            head, attn_weights = self.attention(
                q, k, value, mask=mask, return_attention_weights=True
            )

            heads.append(head)
            attns.append(attn_weights)

        head = torch.stack(heads, dim=2) if self.num_heads > 1 else heads[0]
        attn = torch.stack(attns, dim=2) if self.num_heads > 1 else attns[0]

        head = head.mean(dim=2) if self.num_heads > 1 else head

        output = self.output_layer(head)
        output = output if self.dropout is None else self.dropout(output)

        if return_attn:
            return output, attn
        return output
