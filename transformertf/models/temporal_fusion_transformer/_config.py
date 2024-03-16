from __future__ import annotations

import dataclasses
import typing

from ...config import TransformerBaseConfig


@dataclasses.dataclass
class TemporalFusionTransformerConfig(TransformerBaseConfig):
    n_dim_model: int = 300
    variable_selection_dim: int = 100
    num_heads: int = 4

    num_lstm_layers: int = 2
    dropout: float = 0.1

    output_dim: int = 7

    loss_fn: typing.Literal["mse", "huber", "quantile"] = "quantile"
    prediction_type: typing.Literal["delta", "point"] = "point"
