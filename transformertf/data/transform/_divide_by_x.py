from __future__ import annotations

import logging

import numpy as np
import torch

from ._base import BaseTransform
from ._utils import _as_torch

log = logging.getLogger(__name__)


class DivideByXTransform(BaseTransform):
    _transform_type = BaseTransform.TransformType.XY

    def fit(
        self,
        x: torch.Tensor | np.ndarray | None = None,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> DivideByXTransform:
        if x is not None:
            log.warning(
                "DivideByXTransform does not require fitting, ignoring input data"
            )
        if y is not None:
            log.warning(
                "DivideByXTransform does not require fitting, ignoring target data"
            )

        return self

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        if y is None:
            msg = "y cannot be None for DivideByXTransform"
            raise ValueError(msg)

        x = _as_torch(x)
        y = _as_torch(y)

        return y / x

    def inverse_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        if y is None:
            msg = "y cannot be None for DivideByXTransform"
            raise ValueError(msg)

        x = _as_torch(x)
        y = _as_torch(y)

        return y * x

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
