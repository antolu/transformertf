from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import sklearn.base
import torch
import torch.nn as nn
import tqdm

log = logging.getLogger(__name__)


__all__ = ["BaseTransform", "PolynomialTransform", "RunningNormalizer"]


class BaseTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self) -> None:
        sklearn.base.BaseEstimator.__init__(self)
        sklearn.base.TransformerMixin.__init__(self)

    def inverse_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class PolynomialTransform(nn.Module, BaseTransform):
    """
    Removes a polynomial fit from the data.
    """

    weights: torch.Tensor
    bias: torch.Tensor
    p: torch.Tensor

    def __init__(self, degree: int, num_iterations: int = 1000):
        nn.Module.__init__(self)
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
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray
    ) -> None:
        """
        Fits the polynomial to the data.
        """
        x_tensor = _as_torch(x)
        y_tensor = _as_torch(y)

        self._reset_parameters()

        criterion = nn.MSELoss()

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

    def transform(
        self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        Removes the polynomial fit from the data.
        """
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
            raise ValueError("y cannot be None.")

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


class RunningNormalizer(nn.Module, BaseTransform):
    """
    RunningNormalizer class

    A class for normalizing and scaling data using running statistics.

    Parameters:
        - num_features (int): The number of features in the data. Default is 1.
        - center (torch.Tensor | float): The center value for rescaling. Default is 0.0.
        - scale (torch.Tensor | float): The scale value for rescaling. Default is 1.0.

    Attributes:
        - center_ (torch.Tensor): The running mean center value.
        - scale_ (torch.Tensor): The running standard deviation scale value.
        - n_samples_seen_ (torch.Tensor): The total number of samples seen for fitting.

    Methods:
        - forward(y: torch.Tensor, target_scale: torch.Tensor | None = None) -> torch.Tensor:
            Applies the inverse transformation on the input data.

        - fit(y: torch.Tensor) -> RunningNormalizer:
            Fits the normalizer to the data.

        - transform(y: torch.Tensor, return_scales: bool = False, target_scale: torch.Tensor | None = None)
            -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            Rescales the input data.

        - inverse_transform(y: torch.Tensor) -> torch.Tensor:
            Applies the inverse scaling on the input data.

        - get_parameters() -> torch.Tensor:
            Returns the current parameters of the normalizer.

        - _set_parameters(y_center: torch.Tensor, y_scale: torch.Tensor) -> None:
            Sets the parameters of the normalizer based on the input data.

        - __sklearn_is_fitted__() -> bool:
            Checks if the normalizer has been fitted to the data.

    Note:
        This class inherits from nn.Module, BaseEstimator, and TransformerMixin.
        The `fit` method returns the fitted instance of the normalizer.
        The `transform` method can return the rescaling factors if `return_scales` is set to True.
        The `n_samples_seen_` attribute keeps track of the number of samples seen during fitting.
        The class uses running statistics for incrementally updating the center and scale during fitting.
        The class supports PyTorch tensors for all numeric inputs.

    """

    center_: torch.Tensor
    scale_: torch.Tensor
    n_samples_seen_: torch.Tensor

    def __init__(
        self,
        num_features: int = 1,
        center: torch.Tensor | float = 0.0,
        scale: torch.Tensor | float = 1.0,
    ):
        """
        Parameters
        ----------
        num_features : int
            The number of features in the input data.

        center : torch.Tensor or float, optional
            The center value for normalization. Default is 0.0.

        scale : torch.Tensor or float, optional
            The scale value for normalization. Default is 1.0.

        """
        nn.Module.__init__(self)
        BaseTransform.__init__(self)

        center_ = torch.zeros(num_features, requires_grad=False) + center
        scale_ = torch.zeros(num_features, requires_grad=False) + scale
        n_samples_seen_ = torch.tensor(
            [0], requires_grad=False, dtype=torch.long
        )

        self.register_buffer("center_", center_)
        self.register_buffer("scale_", scale_)
        self.register_buffer("n_samples_seen_", n_samples_seen_)

    def forward(
        self,
        y: np.ndarray | torch.Tensor,
        target_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inverse transform data.
        """
        if target_scale is None:
            target_scale = self.get_parameters()

        center = target_scale[..., 0]
        scale = target_scale[..., 1]

        center = _view_as_y(center, y)
        scale = _view_as_y(scale, y)

        y_unscaled = y * scale + center

        if y.ndim == 1 and y.ndim > target_scale.ndim:
            y_unscaled.squeeze(0)

        return y_unscaled

    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> RunningNormalizer:
        """
        Fit the normalizer to the data.

        :param x: The data to fit the normalizer to.
        :param y: The data to fit the normalizer to.

        :return: The fitted normalizer (self).
        """
        self._set_parameters(y_center=x, y_scale=x)
        return self

    @typing.overload
    def transform(
        self,
        y: torch.Tensor,
        return_scales: typing.Literal[False] = False,
        target_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...

    @typing.overload
    def transform(
        self,
        y: torch.Tensor,
        return_scales: typing.Literal[True],
        target_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def transform(
        self,
        y: torch.Tensor,
        return_scales: bool = False,
        target_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Rescale data.

        :param y: The data to rescale.
        :param return_scales: Whether to return the rescaling factors.
        :param target_scale: The target scale to rescale to.

        :return: The rescaled data.
        """
        y = _as_torch(y)

        if target_scale is None:
            target_scale = self.get_parameters()

        center = target_scale[..., 0]
        scale = target_scale[..., 1]

        center = _view_as_y(center, y)
        scale = _view_as_y(scale, y)

        y_scaled = (y - center) / scale

        if return_scales:
            return y_scaled, target_scale
        else:
            return y_scaled

    def inverse_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Inverse scale.

        :param y: The data to inverse scale.

        :return: The inverse scaled data.
        """
        return self(
            _as_torch(y if y is not None else x),
            target_scale=self.get_parameters(),
        )

    def get_parameters(self) -> torch.Tensor:
        """
        Get the current parameters of the normalizer.

        :return The current parameters of the normalizer.
        """
        return torch.stack([self.center_, self.scale_], dim=-1)

    def _set_parameters(
        self,
        y_center: torch.Tensor,
        y_scale: torch.Tensor,
    ) -> None:
        eps = torch.finfo(y_center.dtype).eps

        dim = tuple(range(y_center.ndim - 1)) if y_center.ndim > 1 else 0
        new_mean = torch.mean(y_center, dim=dim)
        new_scale = torch.std(y_scale, dim=dim) + eps

        n_samples = y_center.shape[0]

        if not self.__sklearn_is_fitted__():
            self.center_ = new_mean
            self.scale_ = new_scale
        else:
            N = self.n_samples_seen_
            self.center_ = (N * self.center_ + n_samples * new_mean) / (
                N + n_samples
            )

            self.scale_ = torch.sqrt(
                (
                    (N - 1) * torch.square(self.scale_)
                    + (n_samples - 1) * torch.square(new_scale)
                )
                / (N + n_samples - 2)
            )

        self.n_samples_seen_ += n_samples

    def __sklearn_is_fitted__(self) -> bool:
        return self.n_samples_seen_.item() > 0


def _view_as_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim < y.ndim:
        return x.view(*(1,) * (y.ndim - x.ndim), *x.size())
    return x


def _as_torch(x: pd.Series | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x
