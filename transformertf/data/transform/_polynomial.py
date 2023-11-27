from __future__ import annotations

from ._base import BaseTransform, TransformType

import numpy as np
import torch
import tqdm

from ._utils import _as_torch


class PolynomialTransform(BaseTransform):
    """
    Removes a polynomial fit from the data.
    """

    _transform_type = TransformType.XY

    weights: torch.Tensor
    bias: torch.Tensor
    p: torch.Tensor

    def __init__(self, degree: int, num_iterations: int = 1000):
        BaseTransform.__init__(self)

        self.degree = degree
        self.num_iterations = num_iterations

        self.p: torch.Tensor

        self.register_buffer("p", torch.arange(1, degree + 1))
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the polynomial component of the data.
        """
        if self.degree == 0:
            return torch.broadcast_to(self.bias, x.shape).to(x)

        xx = x.unsqueeze(-1).pow(self.p.to(x))

        return self.bias.to(x) + self.weights.to(x) @ torch.transpose(
            xx, -1, -2
        )

    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> PolynomialTransform:
        """
        Fits the polynomial to the data.
        """
        if y is None:
            raise ValueError("PolynomialTransform requires y.")
        x_tensor = _as_torch(x)
        y_tensor = _as_torch(y)

        self._reset_parameters()

        criterion = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.99,
        )

        self.train()

        for _ in tqdm.trange(self.num_iterations):
            y_hat = self.forward(x_tensor)

            loss = criterion(y_hat, y_tensor)

            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        self.eval()

        return self

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Removes the polynomial fit from the data.
        """
        if y is None:
            raise ValueError("PolynomialTransform requires y.")

        x_tensor = _as_torch(x)
        y_tensor = _as_torch(y)

        with torch.no_grad():
            return y_tensor - self.forward(x_tensor)

    def inverse_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Adds the polynomial fit back to the data.
        """
        if y is None:
            raise ValueError("PolynomialTransform requires y.")

        x_tensor = _as_torch(x)
        y_tensor = _as_torch(y)

        with torch.no_grad():
            return y_tensor + self.forward(x_tensor)

    def get_derivative(self) -> PolynomialTransform:
        """
        Returns a new PolynomialTransform that represents the derivative of the
        current transform.

        Returns
        -------
        transform : PolynomialTransform
        """
        if self.degree == 0:
            raise ValueError("Cannot get derivative of degree 0 transform.")

        transform = PolynomialTransform(
            degree=self.degree - 1,
            num_iterations=self.num_iterations,
        )
        transform._reset_parameters()

        transform.bias.data = (
            (self.weights[0] * self.p[0]).unsqueeze(0).detach()
        )
        transform.weights.data = (self.weights[1:] * self.p[1:]).detach()

        return transform

    def _reset_parameters(self) -> None:
        weights = torch.zeros(self.degree)
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def __sklearn_is_fitted__(self) -> bool:
        return not bool(
            torch.all(self.weights == 0.0) and torch.all(self.bias == 0.0)
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(degree={self.degree})"


class FixedPolynomialTransform(PolynomialTransform):
    """
    A polynomial transform that is pre-fitted to the data.

    The .fit() method will not do anything.
    """

    def __init__(
        self,
        degree: int,
        weights: list[float] | np.ndarray | torch.Tensor,
        bias: float | np.ndarray | torch.Tensor = torch.zeros(1),
    ):
        super().__init__(degree=degree)
        weights = _as_torch(weights)

        if isinstance(bias, float):
            bias = torch.zeros(1) + bias
        else:
            bias = _as_torch(bias)

        if weights.shape != (degree,):
            raise ValueError(
                f"weights must have shape ({degree},), got {weights.shape}."
            )
        if bias.shape != (1,):
            raise ValueError(f"bias must have shape (1,), got {bias.shape}.")

        self.weights.data = weights
        self.bias.data = bias

    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> PolynomialTransform:
        return self
