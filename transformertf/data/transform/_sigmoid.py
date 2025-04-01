"""
Implementation of a sigmoid transformation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ._base import BaseTransform, _as_torch

log = logging.getLogger(__name__)


class SigmoidTransform(BaseTransform):
    """
    Sigmoid transformation. Does not require fitting, but is controlled only by the
    parameters k and x0.

    The sigmoid function is defined as:

    .. math::
        f(x) = \\frac{1}{1 + e^{-k(x - x0)}}

    Parameters
    ----------
    k : float, default=1.0
        The slope of the sigmoid function.
    x0 : float, default=0.0
        The x value at which the sigmoid function crosses 0.5.

    """

    _transform_type = BaseTransform.TransformType.X

    def __init__(self, k_: float = 1.0, x0_: float = 0.0):
        super().__init__()
        self.k_ = k_
        self.x0_ = x0_

    def fit(
        self,
        x: np.ndarray | torch.Tensor | None = None,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> SigmoidTransform:
        if x is not None:
            msg = "Sigmoid transform does not require fitting, ignoring input data"
            log.warning(msg)
        if y is not None:
            msg = "Sigmoid transform does not require fitting, ignoring target data"
            log.warning(msg)

        return self

    def transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for Sigmoid transform"
            raise ValueError(msg)

        x = _as_torch(x)

        return torch.nn.functional.sigmoid(self.k_ * (x - self.x0_))

    def inverse_transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for Sigmoid transform"
            raise ValueError(msg)

        x = _as_torch(x)

        return self.x0_ + torch.log(x / (1 - x)) / self.k_
