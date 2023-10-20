from __future__ import annotations

import dataclasses
import typing

from ...config import BaseConfig

if typing.TYPE_CHECKING:
    from ._loss import LossWeights


@dataclasses.dataclass
class PhyLSTMConfig(BaseConfig):
    phylstm: typing.Literal[1, 2, 3] | None = 3

    # if tuple, one value per PhyLSTM# module
    hidden_size: int | tuple[int, ...] = 350
    hidden_size_fc: int | tuple[int, ...] | None = 1024
    num_layers: int | tuple[int, ...] = 3
    dropout: float | tuple[float, ...] = 0.2

    lowpass_filter: bool = True
    mean_filter: bool = True

    remove_polynomial: bool = True
    polynomial_degree: int = 1
    polynomial_iterations: int = 1000
    loss_weights: LossWeights | None = None

    input_columns: str = "I_meas_A"
    target_column: str = "B_meas_T"
