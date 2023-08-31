from __future__ import annotations

import logging

import numpy as np
import sklearn.base
import torch
import torch.nn as nn
import tqdm

log = logging.getLogger(__name__)


class PolynomialTransform(
    nn.Module, sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    """
    Removes a polynomial fit from the data.
    """

    weights: torch.Tensor
    bias: torch.Tensor
    p: torch.Tensor

    def __init__(
        self, degree: int, num_iterations: int = 1000, bias: bool = True
    ):
        nn.Module.__init__(self)
        sklearn.base.BaseEstimator.__init__(self)
        sklearn.base.TransformerMixin.__init__(self)

        self.degree = degree
        self.num_iterations = num_iterations
        self._use_bias = bias

        self.p: torch.Tensor

        self.register_buffer("p", torch.arange(1, degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the polynomial component of the data.
        """
        xx = x.unsqueeze(-1).pow(self.p.to(x))

        return self.bias.to(x) + self.weights.to(x) @ torch.transpose(
            xx, -1, -2
        )

    def fit(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray
    ) -> None:
        """
        Fits the polynomial to the data.
        """
        x_tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        y_tensor = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

        self._reset_parameters()

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.train()

        for _ in tqdm.trange(self.num_iterations):
            y_hat = self.forward(x_tensor)

            loss = criterion(y_hat, y_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.eval()

    def transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        Removes the polynomial fit from the data.
        """
        x_tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        y_tensor = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

        return y_tensor - self.forward(x_tensor).detach()

    def inverse_transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        Adds the polynomial fit back to the data.
        """
        x_tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        y_tensor = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

        return y_tensor + self.forward(x_tensor).detach()

    def _reset_parameters(self) -> None:
        weights = torch.zeros(self.degree)
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(torch.zeros(1))

        if not self._use_bias:
            self.bias.requires_grad_(False)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "weights") and hasattr(self, "bias")
