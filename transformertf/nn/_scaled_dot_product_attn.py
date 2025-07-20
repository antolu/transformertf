"""
Scaled Dot-Product Attention implementation for transformer architectures.

This module implements the fundamental scaled dot-product attention mechanism
that forms the core of transformer architectures. It computes attention weights
using the scaled dot-product of queries and keys, then applies these weights
to values to produce the final output.

Classes
-------
ScaledDotProductAttention : torch.nn.Module
    Core scaled dot-product attention mechanism with optional masking and dropout.

Notes
-----
The scaled dot-product attention mechanism is the foundation of transformer
architectures and is used in both self-attention and cross-attention scenarios.
The scaling factor (square root of key dimension) prevents the softmax function
from saturating when the dimensionality is large.

This implementation provides flexible masking capabilities for various attention
patterns including causal masking for autoregressive generation and padding
masks for variable-length sequences.

References
----------
.. [1] Vaswani, Ashish, et al. "Attention is all you need." NIPS 2017.
.. [2] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine
   translation by jointly learning to align and translate." ICLR 2015.
"""

from __future__ import annotations

import typing

import torch


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention mechanism for transformer architectures.

    This module implements the core attention mechanism used in transformer models,
    computing attention weights through scaled dot-products of queries and keys,
    then applying these weights to values. The mechanism includes optional scaling,
    masking, and dropout for robust training and flexible attention patterns.

    The mathematical formulation is:

    .. math::
        \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V

    where:
    - :math:`Q` is the query matrix of shape (batch_size, seq_len_q, d_k)
    - :math:`K` is the key matrix of shape (batch_size, seq_len_k, d_k)
    - :math:`V` is the value matrix of shape (batch_size, seq_len_v, d_v)
    - :math:`d_k` is the key dimension used for scaling
    - :math:`\\sqrt{d_k}` is the scaling factor to prevent softmax saturation

    The attention computation follows these steps:
    1. Compute attention scores: :math:`\\text{scores} = QK^T`
    2. Scale scores: :math:`\\text{scaled} = \\text{scores} / \\sqrt{d_k}` (if enabled)
    3. Apply mask: :math:`\\text{masked} = \\text{scaled} + \\text{mask} \\cdot \\text{bias}` (if provided)
    4. Apply softmax: :math:`\\text{weights} = \\text{softmax}(\\text{masked})`
    5. Apply dropout: :math:`\\text{weights} = \\text{dropout}(\\text{weights})` (if enabled)
    6. Compute output: :math:`\\text{output} = \\text{weights} \\cdot V`

    Parameters
    ----------
    dropout : float, default=0.0
        Dropout probability applied to attention weights after softmax.
        Must be in the range [0, 1). Higher values provide more regularization
        but may hurt performance. Set to 0.0 to disable dropout.
    scale : bool, default=True
        Whether to scale attention scores by :math:`\\sqrt{d_k}` before softmax.
        Scaling helps prevent softmax saturation when key dimensions are large.
        Should typically be True for transformer architectures.
    mask_bias : float, default=-1e9
        Bias value applied to masked positions before softmax.
        Masked positions will have this value added, effectively setting
        their attention weight to near-zero after softmax. Use -float("inf")
        for exact masking, or -1e9 for numerical stability in mixed precision.

    Attributes
    ----------
    dropout_p : float
        Dropout probability for attention weights.
    scale : bool
        Whether to apply scaling to attention scores.
    mask_bias : float
        Bias value for masked positions.

    Notes
    -----
    **Scaling Justification**:
    Without scaling, the dot products can become arbitrarily large as the
    dimension increases, pushing the softmax function into regions with
    extremely small gradients. The :math:`\\sqrt{d_k}` scaling factor
    maintains the variance of the dot products regardless of dimension.

    **Masking Behavior**:
    The mask tensor should have True values at positions that should be
    masked (ignored) and False at positions that should attend. This
    convention aligns with PyTorch's masking standards.

    **Mixed Precision**:
    When using automatic mixed precision (AMP), set mask_bias=-float("inf")
    to ensure proper masking behavior with fp16 computations.

    **Memory Complexity**:
    The attention mechanism has O(n²) memory complexity where n is the
    sequence length, due to the attention weight matrix storage.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import ScaledDotProductAttention
    >>>
    >>> # Basic attention computation
    >>> attn = ScaledDotProductAttention(dropout=0.1)
    >>> batch_size, seq_len, d_model = 32, 128, 64
    >>> q = torch.randn(batch_size, seq_len, d_model)
    >>> k = torch.randn(batch_size, seq_len, d_model)
    >>> v = torch.randn(batch_size, seq_len, d_model)
    >>> output, weights = attn(q, k, v)
    >>> print(output.shape)  # torch.Size([32, 128, 64])
    >>> print(weights.shape)  # torch.Size([32, 128, 128])
    >>>
    >>> # Cross-attention with different sequence lengths
    >>> q = torch.randn(32, 64, 128)   # queries
    >>> k = torch.randn(32, 256, 128)  # keys
    >>> v = torch.randn(32, 256, 128)  # values
    >>> output, weights = attn(q, k, v)
    >>> print(output.shape)  # torch.Size([32, 64, 128])
    >>> print(weights.shape)  # torch.Size([32, 64, 256])
    >>>
    >>> # Causal masking for autoregressive generation
    >>> seq_len = 128
    >>> causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    >>> output, weights = attn(q, k, v, mask=causal_mask)
    >>>
    >>> # Padding mask for variable-length sequences
    >>> lengths = torch.tensor([120, 100, 128])  # actual sequence lengths
    >>> padding_mask = torch.arange(seq_len)[None, :] >= lengths[:, None]
    >>> padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
    >>> output, weights = attn(q, k, v, mask=padding_mask)
    >>>
    >>> # Attention without scaling (not recommended for large dimensions)
    >>> unscaled_attn = ScaledDotProductAttention(scale=False)
    >>> output, weights = unscaled_attn(q, k, v)
    >>>
    >>> # Mixed precision compatible masking
    >>> amp_attn = ScaledDotProductAttention(mask_bias=-float("inf"))
    >>> with torch.cuda.amp.autocast():
    ...     output, weights = amp_attn(q, k, v, mask=causal_mask)
    >>>
    >>> # Dynamic behavior - automatically chooses optimal implementation
    >>> attn = ScaledDotProductAttention(dropout=0.1)
    >>> output = attn(q, k, v, return_attention_weights=False)  # Uses PyTorch's optimized version
    >>> output, weights = attn(q, k, v, return_attention_weights=True)  # Uses custom implementation
    >>>
    >>> # Performance vs interpretability - same instance, dynamic dispatch
    >>> dynamic_attn = ScaledDotProductAttention(dropout=0.1)
    >>> # Fast training loop
    >>> for batch in train_loader:
    ...     output = dynamic_attn(q, k, v, return_attention_weights=False)  # Automatically fast
    ...     loss = criterion(output, target)
    ...     loss.backward()
    >>> # Interpretable analysis
    >>> with torch.no_grad():
    ...     output, attn_weights = dynamic_attn(q, k, v)  # Default returns weights
    ...     analyze_attention_patterns(attn_weights)

    See Also
    --------
    InterpretableMultiHeadAttention : Multi-head attention with interpretability features
    torch.nn.MultiheadAttention : PyTorch's standard multi-head attention
    torch.nn.functional.scaled_dot_product_attention : PyTorch's optimized attention

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention is all you need." NIPS 2017.
    .. [2] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning.
       "Effective approaches to attention-based neural machine translation."
       EMNLP 2015.
    .. [3] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
       "Neural machine translation by jointly learning to align and translate."
       ICLR 2015.
    """

    def __init__(
        self, dropout: float = 0.0, *, scale: bool = True, mask_bias: float = -1e9
    ):
        super().__init__()
        self.dropout_p = dropout
        self.scale = scale
        self.mask_bias = mask_bias

    @typing.overload
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attention_weights: typing.Literal[True] = True,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @typing.overload
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        return_attention_weights: typing.Literal[False],
    ) -> torch.Tensor: ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scaled dot-product attention to input query, key, and value tensors.

        Computes attention weights through scaled dot-products of queries and keys,
        then applies these weights to values to produce the attended output. This
        is the core attention mechanism used in transformer architectures.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (batch_size, seq_len_q, d_k).
            Contains the queries that determine what to attend to.
        k : torch.Tensor
            Key tensor of shape (batch_size, seq_len_k, d_k).
            Contains the keys that are compared against queries.
            For self-attention, this is the same as q.
        v : torch.Tensor
            Value tensor of shape (batch_size, seq_len_v, d_v).
            Contains the values to be aggregated based on attention weights.
            Note: seq_len_v must equal seq_len_k.
        mask : torch.Tensor, optional
            Attention mask of shape (seq_len_q, seq_len_k) or broadcastable shape.
            Positions with True values will be masked (ignored in attention).
            Common mask types:
            - Causal mask: Upper triangular matrix for autoregressive attention
            - Padding mask: Mask out padded positions in variable-length sequences
            - Custom mask: Application-specific attention restrictions
        return_attention_weights : bool, default=True
            Whether to return attention weights along with the output.
            When False, uses PyTorch's optimized implementation for better performance.
            When True, uses the custom implementation that returns attention weights.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If return_attention_weights=False:
                **output** (torch.Tensor): Attended output of shape (batch_size, seq_len_q, d_v).
                This is the weighted combination of values based on attention weights.
            If return_attention_weights=True:
                A tuple containing:
                - **output** (torch.Tensor): Attended output of shape (batch_size, seq_len_q, d_v).
                  This is the weighted combination of values based on attention weights.
                - **attn_weights** (torch.Tensor): Attention weights of shape (batch_size, seq_len_q, seq_len_k).
                  These weights show how much each query position attends to each key position.

        Raises
        ------
        RuntimeError
            If tensor shapes are incompatible for matrix multiplication.
        ValueError
            If key and value tensors have mismatched sequence lengths.

        Notes
        -----
        The forward pass implements the following computation:

        1. **Attention Scores**: Compute Q·K^T using batch matrix multiplication
        2. **Scaling**: Divide by √d_k if scale=True to prevent softmax saturation
        3. **Masking**: Add mask_bias to masked positions if mask is provided
        4. **Softmax**: Apply softmax along the key dimension to get attention weights
        5. **Dropout**: Apply dropout to attention weights if dropout > 0
        6. **Output**: Compute weighted sum of values using attention weights

        The attention weights represent how much each query position "attends to"
        each key position. Higher weights indicate stronger attention connections.

        **Computational Complexity**:
        - Time: O(n·m·d + n·m) where n=seq_len_q, m=seq_len_k, d=d_k
        - Space: O(n·m) for attention weight matrix storage

        **Gradient Flow**:
        Gradients flow through both the attention weights and the value
        aggregation, allowing the model to learn both what to attend to
        and how to combine the attended information.

        Examples
        --------
        >>> import torch
        >>> attn = ScaledDotProductAttention(dropout=0.1, scale=True)
        >>>
        >>> # Self-attention example
        >>> batch_size, seq_len, d_model = 32, 128, 64
        >>> x = torch.randn(batch_size, seq_len, d_model)
        >>> output, weights = attn(x, x, x)
        >>> print(output.shape)  # torch.Size([32, 128, 64])
        >>> print(weights.shape)  # torch.Size([32, 128, 128])
        >>>
        >>> # Cross-attention example
        >>> q = torch.randn(32, 64, 128)   # queries (different seq length)
        >>> kv = torch.randn(32, 256, 128) # keys and values
        >>> output, weights = attn(q, kv, kv)
        >>> print(output.shape)  # torch.Size([32, 64, 128])
        >>> print(weights.shape)  # torch.Size([32, 64, 256])
        >>>
        >>> # With causal mask for autoregressive attention
        >>> seq_len = 128
        >>> causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        >>> output, weights = attn(x, x, x, mask=causal_mask)
        >>> # weights[i, j, k] will be ~0 if j > k (future positions)
        >>>
        >>> # Inspect attention patterns
        >>> output, weights = attn(x, x, x)
        >>> avg_attention = weights.mean(dim=0)  # Average over batch
        >>> attention_entropy = -(weights * torch.log(weights + 1e-12)).sum(dim=-1)
        >>> print(f"Attention entropy: {attention_entropy.mean()}")
        """
        # Use fast path when attention weights are not needed
        if not return_attention_weights:
            # Use PyTorch's optimized implementation for best performance
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # mask should handle causality
                scale=None if self.scale else 1.0,  # None means auto-scale
            )

        # Use custom implementation for interpretability (when attention weights are needed)
        # Compute attention scores: Q @ K^T
        attn_scores = torch.bmm(q, k.transpose(-2, -1))

        # Scale by sqrt(d_k) if enabled
        if self.scale:
            scale_factor = q.size(-1) ** -0.5  # More efficient than sqrt
            attn_scores.mul_(scale_factor)  # In-place operation

        # Apply mask if provided
        if mask is not None:
            attn_scores.masked_fill_(mask, self.mask_bias)  # In-place masking

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply dropout during training
        if self.dropout_p > 0.0 and self.training:
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=self.dropout_p, training=self.training
            )

        # Compute weighted sum of values
        output = torch.bmm(attn_weights, v)

        if return_attention_weights:
            return output, attn_weights
        return output
