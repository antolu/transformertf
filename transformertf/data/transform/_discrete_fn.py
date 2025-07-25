from __future__ import annotations

import os
import pathlib

import numpy as np
import torch

from ._base import BaseTransform
from ._utils import _as_numpy, _as_torch


def parse_csv_to_numpy_array(csv_string: str, skip_rows: int = 0) -> np.ndarray:
    # Split the string into rows
    rows = csv_string.strip().split("\n")
    rows = rows[skip_rows:]

    # Split each row into values and convert to float
    data = [list(map(float, row.split(","))) for row in rows]

    # Convert to numpy array
    return np.array(data)


class DiscreteFunctionTransform(BaseTransform):
    """
    A discrete function is a function that is defined by a set of points
    (x, y) where x is strictly ascending. The function is defined by
    interpolating between the points.

    The interpolation is done using the `numpy.interp` function, and therefore
    any torch tensors are converted to numpy arrays and then back to torch
    tensors.
    """

    _transform_type = BaseTransform.TransformType.XY

    def __init__(
        self,
        xs_: np.ndarray | torch.Tensor | os.PathLike | str,
        ys_: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__()

        if isinstance(xs_, os.PathLike | str):
            xs_, ys_ = self.parse_csv_to_numpy_array(xs_, skip_rows=1)
        elif ys_ is None:
            msg = "DiscreteFunctionTransform requires y."
            raise ValueError(msg)

        self.register_buffer("xs_", _as_torch(xs_))
        self.register_buffer("ys_", _as_torch(ys_))

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
        xs = _as_numpy(self.xs_)
        ys = _as_numpy(self.ys_)
        return _as_torch(np.interp(x, xs, ys))

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

    @staticmethod
    def parse_csv_to_numpy_array(
        csv_path: os.PathLike | str, skip_rows: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(csv_path, os.PathLike | str):
            csv_path = pathlib.Path(csv_path).expanduser()
            csv_path = os.fspath(csv_path)
        data = np.loadtxt(csv_path, skiprows=skip_rows, delimiter=",")
        return data[:, 0], data[:, 1]

    @classmethod
    def from_csv(cls, csv_path: os.PathLike | str) -> DiscreteFunctionTransform:
        return cls(*cls.parse_csv_to_numpy_array(csv_path, skip_rows=1))

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
