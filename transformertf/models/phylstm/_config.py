from __future__ import annotations

import dataclasses
import typing

from ...config import TimeSeriesBaseConfig

if typing.TYPE_CHECKING:
    from ._loss import LossWeights


@dataclasses.dataclass
class PhyLSTMConfig(TimeSeriesBaseConfig):
    """PhyLSTM specific configuration"""

    phylstm: typing.Literal[1, 2, 3] | None = 3

    # network parameters
    # if tuple, one value per PhyLSTM# module
    hidden_size: int | tuple[int, ...] = 350
    hidden_size_fc: int | tuple[int, ...] | None = 1024
    num_layers: int | tuple[int, ...] = 3
    dropout: float | tuple[float, ...] = 0.2

    # preprocessing parameters
    lowpass_filter: bool = True
    mean_filter: bool = True

    # specific PhyLSTM configuration
    loss_weights: LossWeights | None = None

    # override defaults
    remove_polynomial: bool = True
    known_covariates_cols: str = "I_meas_A"
    target_col: str = "B_meas_T"
