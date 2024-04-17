from __future__ import annotations

import numpy as np
import torch

from ._base import BaseTransform, TransformType
from ._utils import _as_numpy, _as_torch


class DiscreteFunctionTransform(BaseTransform):
    """
    A discrete function is a function that is defined by a set of points
    (x, y) where x is strictly ascending. The function is defined by
    interpolating between the points.

    The interpolation is done using the `numpy.interp` function, and therefore
    any torch tensors are converted to numpy arrays and then back to torch
    tensors.
    """

    _transform_type = TransformType.XY

    def __init__(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor):
        super().__init__()
        self.xs = _as_numpy(x)
        self.ys = _as_numpy(y)

    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> DiscreteFunctionTransform:
        return self

    def forward(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Computes the discrete function at the given x values.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor

        Returns
        -------
        torch.Tensor
            The discrete function evaluated at the given x values.
        """
        x = _as_numpy(x)
        return _as_torch(np.interp(x, self.xs, self.ys))

    def transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Evaluates the discrete function at the given x values and subtracts
        the result from y (i.e. y - f(x)).

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
        y : np.ndarray | torch.Tensor
            Not optional, as the function is defined by the points (x, y).

        Returns
        -------
        torch.Tensor
            y - f(x)
        """
        if y is None:
            msg = "DiscreteFunction requires y."
            raise ValueError(msg)

        y_tensor = _as_torch(y)

        with torch.no_grad():
            return y_tensor - self.forward(x)

    def inverse_transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            msg = "DiscreteFunctionTransform requires y."
            raise ValueError(msg)

        y_tensor = _as_torch(y)

        with torch.no_grad():
            return y_tensor + self.forward(x)

    @classmethod
    def from_csv(cls, csv_path: str) -> DiscreteFunctionTransform:
        data = np.loadtxt(csv_path, skiprows=2, delimiter=",")
        return cls(data[:, 0], data[:, 1])

    def inverse_function(self) -> DiscreteFunctionTransform:
        """
        Returns a new DiscreteFunctionTransform that is the equivalent of the
        inverse of function.

        Returns
        -------
        DiscreteFunctionTransform
            The inverse of the function.
        """
        return DiscreteFunctionTransform(self.ys, self.xs)

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return True
