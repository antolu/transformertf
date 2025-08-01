from __future__ import annotations

import torch

from ...nn import MLP
from ._pos_enc import SimplePositionalEncoding

__all__ = ["VanillaTransformerModel"]


class VanillaTransformerModel(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        out_seq_len: int | None = None,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
    ):
        super().__init__()
        if out_seq_len is None:
            out_seq_len = seq_len

        self.num_features = num_features
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.activation = activation
        self.fc_dim = fc_dim

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
            hidden_dim=self.fc_dim,
            output_dim=output_dim,
            dropout=self.dropout,
            activation=self.activation,  # type: ignore[arg-type]
        )  # [bs, seq_len, output_dim]

        self.src_mask: torch.Tensor
        self.tgt_mask: torch.Tensor
        self.register_buffer("src_mask", generate_mask(self.seq_len))
        self.register_buffer("tgt_mask", generate_mask(self.out_seq_len))

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_embedding(source)
        x = self.pos_encoder(x)

        t = self.feature_embedding(target)
        t = self.pos_encoder(t)

        if src_mask is None:
            src_mask = self.src_mask
        if tgt_mask is None:
            tgt_mask = self.tgt_mask

        decoding = self.transformer(x, t, src_mask=src_mask, tgt_mask=tgt_mask)

        return self.fc(decoding)


def generate_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))
