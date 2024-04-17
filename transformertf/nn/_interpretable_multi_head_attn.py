from __future__ import annotations

import typing

import torch


class InterpretableMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        n_dim_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        """
        Interpretable Multi-Head Attention module, based on the equation:

        .. math::
            \\text{InterpretableMultiHeadAttention}(Q, K, V) = \\frac{1}{h} \\sum_{i=1}^h \\text{head}_i W_H

        where :math:`\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW^V)`.

        Parameters
        ----------
        num_heads: int
            Number of attention heads
        n_dim_model: int
            Dimension of the model
        dropout: float, default=0.1
            Dropout rate
        activation: str, default="relu"
            Activation function
        """
        super().__init__()
        self.num_heads = n_heads
        self.d_model = n_dim_model
        self.dropout = dropout

        self.d_q = self.d_k = self.d_v = n_dim_model // n_heads

        self.query_layers = torch.nn.ModuleList([
            torch.nn.Linear(n_dim_model, self.d_q) for _ in range(n_heads)
        ])
        self.key_layers = torch.nn.ModuleList([
            torch.nn.Linear(n_dim_model, self.d_k) for _ in range(n_heads)
        ])
        self.value_layer = torch.nn.Linear(n_dim_model, self.d_v)

        self.output_layer = torch.nn.Linear(self.d_v, n_dim_model)

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
        Forward pass.

        Parameters
        ----------
        query: torch.Tensor
            Query tensor
        key: torch.Tensor
            Key tensor
        value: torch.Tensor
            Value tensor
        mask: torch.Tensor, optional
            Mask tensor, by default None

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        value = self.value_layer(value)

        heads = []
        attns = []
        for i in range(self.num_heads):
            q = self.query_layers[i](query)
            k = self.key_layers[i](key)
            v = torch.eye(
                value.shape[1], device=query.device
            )  # hack to get attention scores

            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout
            )
            head = attn @ value
            attns.append(attn)
            heads.append(head)

        head = torch.stack(heads, dim=2) if self.num_heads > 1 else heads[0]
        attn = torch.stack(attns, dim=2) if self.num_heads > 1 else attns[0]

        head = head.mean(dim=2) if self.num_heads > 1 else head

        output = self.output_layer(head)
        output = torch.nn.functional.dropout(output, p=self.dropout)

        return output, attn
