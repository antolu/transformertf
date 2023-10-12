from __future__ import annotations

from ...config import BaseConfig


class PreisachConfig(BaseConfig):
    """Configuration for Preisach model."""

    polynomial_degree: int = 1
    polynomial_bias: bool = True

    current_column: str = "I_meas_A"
    field_column: str = "B_meas_T"
