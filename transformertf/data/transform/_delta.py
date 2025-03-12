from __future__ import annotations

import numpy as np
import torch

from ._base import BaseTransform, _as_torch


class DeltaTransform(BaseTransform):
    """
    Transform to calculate the delta of a time series.
    The :meth:`transform` method calculates the difference between consecutive
    elements of the input data.
    """

    _transform_type = BaseTransform.TransformType.X

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the delta of the input data.
        Prepends a zero to the result.

        Parameters
        ----------
        y : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The delta of the input data.
        """
        diff = y[1:] - y[:-1]
        mean = diff.mean()
        return torch.cat([diff, mean.unsqueeze(0)], dim=0)

    def fit(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> DeltaTransform:
        return self

    def transform(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Calculate the delta of the input data.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The input data.

        y : np.ndarray or torch.Tensor, optional
            Ignored.

        Returns
        -------
        torch.Tensor
            The delta of the input data.
        """
        if y is None:
            y = x

        y = _as_torch(y)
        return self.forward(y)

    def inverse_transform(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Calculate the inverse delta of the input data.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The input data.

        y : np.ndarray or torch.Tensor, optional
            Ignored.

        Returns
        -------
        torch.Tensor
            The inverse delta of the input data.
        """
        if y is None:
            y = x

        y = _as_torch(y)
        return torch.cumsum(y, dim=0)

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return True
