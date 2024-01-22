from __future__ import annotations

import dataclasses
import typing

from ...config import TransformerBaseConfig


@dataclasses.dataclass
class TSMixerConfig(TransformerBaseConfig):
    fc_dim: int = 512
    num_blocks: int = 4

    num_static_features: int = 0
    hidden_dim: int | None = None
    """ Hidden dim of TSMixer blocks, if None, defaults to num_features. """

    dropout: float = 0.1
    activation: str = "relu"
    norm: typing.Literal["batch", "layer"] = "batch"

    loss_fn: typing.Literal["mse", "quantile"] = "mse"
