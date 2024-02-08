from __future__ import annotations

import dataclasses

from ...config import TransformerBaseConfig


@dataclasses.dataclass
class TransformerEncoderConfig(TransformerBaseConfig):
    n_dim_model: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 6
    dropout: float = 0.1

    activation: str = "relu"

    fc_dim: int = 1024
    output_dim: int = 7
