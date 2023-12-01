from __future__ import annotations

import dataclasses

from ...config import TimeSeriesBaseConfig


@dataclasses.dataclass
class TSMixerConfig(TimeSeriesBaseConfig):
    dropout: float = 0.1

    activation: str = "relu"

    fc_dim: int = 1024
    output_dim: int = 7
