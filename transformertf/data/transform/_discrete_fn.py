from __future__ import annotations

import os
import pathlib
import typing

import numpy as np
import torch

from ._base import BaseTransform, TransformType
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

    _transform_type = TransformType.XY

    def __init__(
        self,
        xs: np.ndarray | torch.Tensor,
        ys: np.ndarray | torch.Tensor,
    ):
        super().__init__()

        self.xs = _as_numpy(xs)
        self.ys = _as_numpy(ys)

    def __new__(
        cls,
        xs: np.ndarray | torch.Tensor | os.PathLike | str,
        ys: np.ndarray | torch.Tensor | None = None,
    ) -> typing.Self:
        if isinstance(xs, os.PathLike | str):
            return cls.from_csv(xs)

        if ys is None:
            msg = "DiscreteFunction requires ys if xs is not a path."
            raise ValueError(msg)

        return super().__new__(cls)

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
    def from_csv(cls, csv_path: os.PathLike | str) -> DiscreteFunctionTransform:
        if isinstance(csv_path, os.PathLike | str):
            csv_path = pathlib.Path(csv_path).expanduser()
            csv_path = os.fspath(csv_path)
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
