from __future__ import annotations

from ...config import TimeSeriesBaseConfig


class PreisachConfig(TimeSeriesBaseConfig):
    """Configuration for Preisach model."""

    # set defaults
    polynomial_degree: int = 1
    current_column: str = "I_meas_A"
    field_column: str = "B_meas_T"
