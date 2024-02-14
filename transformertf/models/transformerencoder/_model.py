from __future__ import annotations

import torch

from ...nn import MLP
from ...utils import ACTIVATIONS
from ..transformer import SimplePositionalEncoding
from ..transformer._model import generate_mask

__all__ = ["TransformerEncoder"]


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        n_dim_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        activation: ACTIVATIONS = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
    ):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.n_dim_model = n_dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        self.fc_dim = fc_dim

        self.feature_embedding = torch.nn.Linear(
            self.num_features, self.n_dim_model
        )  # [bs, seq_len, n_dim_model]
        self.pos_encoder = SimplePositionalEncoding(
            dim_model=self.n_dim_model, dropout=self.dropout
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.n_dim_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.num_encoder_layers
        )
        self.fc = MLP(
            input_dim=self.n_dim_model,
            hidden_dim=self.fc_dim,
            output_dim=output_dim,
            dropout=self.dropout,
            activation=self.activation,
        )  # [bs, seq_len, output_dim]

        self.src_mask: torch.Tensor
        self.register_buffer("src_mask", generate_mask(self.seq_len))

    def forward(
        self,
        source: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_embedding(source)
        x = self.pos_encoder(x)

        encoded = self.transformer(x)

        out = self.fc(encoded)

        return out
