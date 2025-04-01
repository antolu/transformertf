"""
Implementation of a sigmoid transformation.
"""

from __future__ import annotations

import torch

from ._base import BaseTransform


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

    def __init__(self, k: float = 1.0, x0: float = 0.0):
        super().__init__()
        self.k = k
        self.x0 = x0

    def fit(
        self,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> SigmoidTransform:
        if x is not None:
            msg = "Sigmoid transform does not require fitting, ignoring input data"
            raise ValueError(msg)
        if y is not None:
            msg = "Sigmoid transform does not require fitting, ignoring target data"
            raise ValueError(msg)

        return self

    def transform(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for Sigmoid transform"
            raise ValueError(msg)

        return torch.nn.functional.sigmoid(self.k * (x - self.x0))

    def inverse_transform(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for Sigmoid transform"
            raise ValueError(msg)

        return self.x0 + torch.log(x / (1 - x)) / self.k
