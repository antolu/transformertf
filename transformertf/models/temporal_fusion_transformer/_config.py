from __future__ import annotations

import dataclasses

from ...config import TransformerBaseConfig


@dataclasses.dataclass
class TemporalFusionTransformerConfig(TransformerBaseConfig):
    n_dim_model: int = 300
    hidden_continuous_dim: int = 8
    num_heads: int = 4

    num_lstm_layers: int = 2
    dropout: float = 0.1

    output_dim: int = 7
