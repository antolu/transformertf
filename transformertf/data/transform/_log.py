"""
Implementation of a Logarithm transformation.

The Logarithm transformation used to transform the
data by taking the natural logarithm of the data.
"""

from __future__ import annotations

import numpy as np
import torch

from ._base import BaseTransform, TransformType


class LogTransform(BaseTransform):
    """
    A Logarithm transformation is a transformation that
    takes the natural logarithm of the data.
    """

    _transform_type = TransformType.X

    def fit(
        self,
        x: torch.Tensor | np.ndarray | None = None,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> LogTransform:
        return self

    def transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        return torch.log(x)

    def inverse_transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        return torch.exp(x)

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return True
