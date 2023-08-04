from __future__ import annotations


import dataclasses
from ...config import BaseConfig
import typing


if typing.TYPE_CHECKING:
    from ._loss import LossWeights


@dataclasses.dataclass
class PhyLSTMConfig(BaseConfig):
    phylstm: typing.Literal[1, 2, 3] | None = 3
    hidden_size: int = 350
    num_layers: int = 3
    dropout: float = 0.2

    lowpass_filter: bool = True
    mean_filter: bool = True

    loss_weights: LossWeights | None = None
