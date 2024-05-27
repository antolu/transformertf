from __future__ import annotations

import typing

import torch

from ...nn import MLP, GatedLinearUnit, GatedResidualNetwork
from ..transformer._model import generate_mask
from ..transformer._pos_enc import SimplePositionalEncoding

__all__ = ["TransformerV2Model"]


class LSTMEmbedding(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=output_dim,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = GatedLinearUnit(output_dim)
        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.norm(x)


class TransformerV2Model(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        out_seq_len: int | None = None,
        n_dim_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        embedding: typing.Literal["mlp", "lstm"] = "mlp",
        output_dim: int = 7,  # quantile loss
    ):
        super().__init__()
        if out_seq_len is None:
            out_seq_len = seq_len

        self.num_features = num_features
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.n_dim_model = n_dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.activation = activation
        self.fc_dim = fc_dim

        if embedding == "lstm":
            self.feature_embedding = LSTMEmbedding(
                input_dim=self.num_features,
                output_dim=self.n_dim_model,
                dropout=self.dropout,
            )
        elif embedding == "mlp":
            self.feature_embedding = torch.nn.Linear(
                self.num_features, self.n_dim_model
            )  # [bs, seq_len, n_dim_model]
        else:
            msg = f"embedding model: {embedding} not supported"
            raise ValueError(msg)

        self.pos_encoder = SimplePositionalEncoding(
            dim_model=self.n_dim_model, dropout=self.dropout
        )
        self.transformer = torch.nn.Transformer(
            d_model=self.n_dim_model,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            dim_feedforward=fc_dim,
        )
        self.grn3 = GatedResidualNetwork(
            input_dim=self.n_dim_model,
            hidden_dim=(
                self.fc_dim if isinstance(self.fc_dim, int) else self.fc_dim[0]
            ),
            output_dim=self.n_dim_model,
            dropout=self.dropout,
            activation=self.activation,  # type: ignore[arg-type]
        )
        self.glu3 = GatedLinearUnit(self.n_dim_model)
        self.norm3 = torch.nn.LayerNorm(self.n_dim_model)
        self.fc = MLP(
            input_dim=self.n_dim_model,
            hidden_dim=None,
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
        target_emb = t
        t = self.pos_encoder(t)

        if src_mask is None:
            src_mask = self.src_mask
        if tgt_mask is None:
            tgt_mask = self.tgt_mask

        decoding = self.transformer(x, t, src_mask=src_mask, tgt_mask=tgt_mask)
        decoding = self.grn3(decoding)
        decoding = self.glu3(decoding)
        decoding = self.norm3(decoding + target_emb)

        return self.fc(decoding)
