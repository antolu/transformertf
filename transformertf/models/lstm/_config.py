from __future__ import annotations

import dataclasses
import typing

from ...config import TimeSeriesBaseConfig


@dataclasses.dataclass
class LSTMConfig(TimeSeriesBaseConfig):
    """LSTM specific configuration"""

    # if tuple, one value per PhyLSTM# module
    hidden_size: int = 350
    hidden_size_fc: int = 1024
    num_layers: int = 3
    dropout: float = 0.2

    loss_fn: typing.Literal["mse", "huber", "mae", "quantile"] = "mse"
