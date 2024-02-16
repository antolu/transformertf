from __future__ import annotations

import dataclasses
import typing

from ...config import TransformerBaseConfig


@dataclasses.dataclass
class TransformerV2Config(TransformerBaseConfig):
    n_dim_model: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1

    embedding: typing.Literal["mlp", "lstm"] = "mlp"
    lstm_hidden_dim: int = 128

    activation: str = "relu"

    fc_dim: int | tuple[int, ...] = 1024
    output_dim: int = 7

    loss_fn: typing.Literal["mse", "huber", "quantile"] = "quantile"
