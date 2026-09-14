from __future__ import annotations

import torch

from ...nn import MLP
from ._pos_enc import SimplePositionalEncoding

__all__ = ["VanillaTransformerModel"]


class VanillaTransformerModel(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        d_fc: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
        causal_attention: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.activation = activation
        self.d_fc = d_fc
        self.causal_attention = causal_attention

        self.feature_embedding = torch.nn.Linear(
            self.num_features, self.d_model
        )  # [bs, seq_len, d_model]
        self.pos_encoder = SimplePositionalEncoding(
            dim_model=self.d_model, dropout=self.dropout
        )
        self.transformer = torch.nn.Transformer(
            d_model=self.d_model,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.fc = MLP(
            input_dim=self.d_model,
            d_hidden=self.d_fc,
            output_dim=output_dim,
            dropout=self.dropout,
            activation=self.activation,  # type: ignore[arg-type]
        )  # [bs, seq_len, output_dim]

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_embedding(source)
        x = self.pos_encoder(x)

        t = self.feature_embedding(target)
        t = self.pos_encoder(t)

        if self.causal_attention and tgt_mask is None:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                target.size(1), device=target.device
            )

        decoding = self.transformer(
            x, t, tgt_mask=tgt_mask, tgt_is_causal=self.causal_attention
        )

        return self.fc(decoding)


def generate_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))
