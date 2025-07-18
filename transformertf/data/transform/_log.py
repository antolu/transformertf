"""
Implementation of a Logarithm transformation.

The Logarithm transformation used to transform the
data by taking the natural logarithm of the data.
"""

from __future__ import annotations

import numpy as np
import torch

from ._base import BaseTransform
from ._utils import _as_torch


class LogTransform(BaseTransform):
    """
    A Logarithm transformation is a transformation that
    takes the natural logarithm of the data.
    """

    _transform_type = BaseTransform.TransformType.X
    bias_: torch.Tensor

    def __init__(self, bias_: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("bias_", torch.tensor(bias_))

    def fit(
        self,
        x: torch.Tensor | np.ndarray | None = None,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> LogTransform:
        return self

    def transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        return torch.log(_as_torch(x) + self.bias_)

    def inverse_transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        return torch.exp(_as_torch(x)) - self.bias_

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return True

    def __str__(self) -> str:
        params = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k.endswith("_")
        }
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in params.items())})"


class Log1pTransform(BaseTransform):
    """
    A Logarithm transformation is a transformation that
    takes the natural logarithm of the data plus one.
    """

    _transform_type = BaseTransform.TransformType.X

    def fit(
        self,
        x: torch.Tensor | np.ndarray | None = None,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> Log1pTransform:
        return self

    def transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        return torch.log1p(_as_torch(x))

    def inverse_transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> torch.Tensor:
        return torch.expm1(_as_torch(x))

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
