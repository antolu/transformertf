from __future__ import annotations

import logging
import typing

import numpy as np
import torch

from ._base import BaseTransform
from ._utils import _as_torch, _view_as_y

log = logging.getLogger(__name__)


class RunningNormalizer(BaseTransform):
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

    _transform_type = BaseTransform.TransformType.X

    center_: torch.Tensor
    scale_: torch.Tensor
    n_samples_seen_: torch.Tensor

    def __init__(
        self,
        num_features_: int = 1,
        center_: torch.Tensor | float = 0.0,
        scale_: torch.Tensor | float = 1.0,
        n_samples_seen_: torch.Tensor | float = 0,
        *,
        frozen_: bool = False,
    ):
        """
        Parameters
        ----------
        num_features_ : int
            The number of features in the input data.

        center_ : torch.Tensor or float, optional
            The center value for normalization. Default is 0.0.

        scale_ : torch.Tensor or float, optional
            The scale value for normalization. Default is 1.0.

        n_samples_seen_ : torch.Tensor or float, optional
            The total number of samples seen for fitting. Default is 0.

        frozen_ : bool, optional
            Whether to freeze the normalizer. Default is False.

        Notes
        -----
        The arguments end with a `_` for scipy to detect them properly.
        """
        super().__init__()

        center_ += torch.zeros(num_features_, requires_grad=False)
        scale_ += torch.zeros(num_features_, requires_grad=False)
        n_samples_seen_ += torch.zeros(
            num_features_, requires_grad=False, dtype=torch.long
        )

        self.register_buffer(
            "num_features_", torch.tensor(num_features_, requires_grad=False)
        )
        self.register_buffer("center_", center_)
        self.register_buffer("scale_", scale_)
        self.register_buffer("n_samples_seen_", n_samples_seen_)
        self.register_buffer("frozen_", torch.tensor(frozen_, requires_grad=False))

    def forward(
        self,
        y: np.ndarray | torch.Tensor,
        target_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inverse transform data.
        """
        y = _as_torch(y)
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
        if self.frozen_:
            log.warning("The normalizer is frozen and cannot be fitted.")
            return self
        x = _as_torch(x)

        self._set_parameters(y_center=x, y_scale=x)
        return self

    @typing.overload
    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: None = None,
        target_scale: torch.Tensor | None = None,
        *,
        return_scales: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
        target_scale: torch.Tensor | None = None,
        *,
        return_scales: typing.Literal[True] = True,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
        target_scale: torch.Tensor | None = None,
        *,
        return_scales: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Rescale data.

        :param x: The data to rescale.
        :param y: The data to rescale.
        :param return_scales: Whether to return the rescaling factors.
        :param target_scale: The target scale to rescale to.

        :return: The rescaled data.
        """
        if y is None:
            y = x
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
        new_mean = torch.mean(y_center, dim=dim, keepdim=True)
        new_scale = torch.std(y_scale, dim=dim, keepdim=True) + eps

        n_samples = y_center.shape[0]

        if not self.__sklearn_is_fitted__():
            self.center_ = new_mean
            self.scale_ = new_scale
        else:
            N = self.n_samples_seen_
            self.center_ = (N * self.center_ + n_samples * new_mean) / (N + n_samples)

            self.scale_ = torch.sqrt(
                (
                    (N - 1) * torch.square(self.scale_)
                    + (n_samples - 1) * torch.square(new_scale)
                )
                / (N + n_samples - 2)
            )

        self.n_samples_seen_ += n_samples

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return torch.all(self.n_samples_seen_ > 0).item() or self.frozen_

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"


class MinMaxScaler(BaseTransform):
    _transform_type = BaseTransform.TransformType.X
    min_: torch.Tensor
    max_: torch.Tensor
    data_min_: torch.Tensor
    data_max_: torch.Tensor
    frozen_: torch.Tensor
    num_features_: torch.Tensor

    def __init__(
        self,
        num_features_: int = 1,
        min_: torch.Tensor | float = 0.0,
        max_: torch.Tensor | float = 1.0,
        data_min_: torch.Tensor | float = 0.0,
        data_max_: torch.Tensor | float = 1.0,
        *,
        frozen_: bool = False,
    ):
        super().__init__()
        self.register_buffer("min_", torch.ones(num_features_) * min_)
        self.register_buffer("max_", torch.ones(num_features_) * max_)

        # add data min and max
        self.register_buffer("data_min_", torch.ones(num_features_) * data_min_)
        self.register_buffer("data_max_", torch.ones(num_features_) * data_max_)

        self.register_buffer("frozen_", torch.tensor(frozen_, requires_grad=False))
        self.register_buffer(
            "num_features_", torch.tensor(num_features_, requires_grad=False)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward transformation to the input data.

        Parameters
        ----------
        y : torch.Tensor
            The input data to transform.

        Returns
        -------
        torch.Tensor
            The transformed data.
        """
        return (y - self.data_min_) / (self.data_max_ - self.data_min_) * (
            self.max_ - self.min_
        ) + self.min_

    def fit(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> MinMaxScaler:
        """
        Fit the scaler to the data.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The data to fit the scaler to.
        y : np.ndarray or torch.Tensor, optional
            The data to fit the scaler to. If None, x is used.
            Default is None.

        Returns
        -------
        MinMaxScaler
            The fitted scaler.
        """
        if self.frozen_:
            log.warning("The scaler is frozen and cannot be fitted.")
            return self

        if y is None:
            y = x

        y = _as_torch(y)
        self.data_max_ = torch.max(y, dim=0)[0].view_as(self.data_max_)
        self.data_min_ = torch.min(y, dim=0)[0].view_as(self.data_min_)

        if torch.allclose(self.data_max_, self.data_min_):
            msg = "The data has no variance. The scaler will not be able to transform the data."
            log.warning(msg)

        return self

    def transform(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The data to transform.
        y : np.ndarray or torch.Tensor, optional
            The data to transform. If None, x is used.
            Default is None.

        Returns
        -------
        torch.Tensor
            The transformed data.
        """
        if y is None:
            y = x
        return self.forward(_as_torch(y))

    def inverse_transform(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Inverse transform the data using the fitted scaler.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The data to inverse transform.
        y : np.ndarray or torch.Tensor, optional
            The data to inverse transform. If None, x is used.
            Default is None.

        Returns
        -------
        torch.Tensor
            The inverse transformed data.
        """
        if y is None:
            y = x
        return _as_torch(y) * (self.data_max_ - self.data_min_) + self.data_min_

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return (
            not (
                torch.all(self.data_max_ == 1.0).item()
                and torch.all(self.data_min_ == 0.0).item()
            )
            or self.frozen_
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaxScaler(MinMaxScaler):
    """
    A class used to scale data based on provided maximum and minimum values.
    The data is only scaled w.r.t. the maximum value (i.e. the scaled values can be negative).

    Attributes
    ----------
    num_features_ : int
        The number of features to scale.
    max_ : torch.Tensor or float
        The maximum value for scaling. Stored as a buffer.
    data_max_ : torch.Tensor or float
        The maximum value of the data. Stored as a buffer.

    Parameters
    ----------
    num_features_ : int, optional
        The number of features to scale (default is 1).
    max_ : torch.Tensor or float, optional
        The maximum value for scaling (default is 1.0).
    data_max_ : torch.Tensor or float, optional
        The maximum value of the data (default is 1.0).
    """

    _transform_type = BaseTransform.TransformType.X
    max_: torch.Tensor
    data_max_: torch.Tensor
    num_features_: torch.Tensor
    frozen_: torch.Tensor

    def __init__(
        self,
        num_features_: int = 1,
        max_: torch.Tensor | float = 1.0,
        data_max_: torch.Tensor | float = 1.0,
        *,
        frozen_: bool = False,
    ):
        super().__init__()
        max_ += torch.zeros(num_features_, requires_grad=False)
        self.register_buffer("max_", max_)
        self.register_buffer(
            "data_max_", data_max_ + torch.zeros(num_features_, requires_grad=False)
        )
        self.register_buffer("frozen_", torch.tensor(frozen_, requires_grad=False))
        self.register_buffer(
            "num_features_", torch.tensor(num_features_, requires_grad=False)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # type: ignore[has-type]
        return y / self.data_max_ * self.max_

    def fit(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> MaxScaler:
        """
        Fit the scaler to the data.



        """
        if self.frozen_:
            log.warning("The scaler is frozen and cannot be fitted.")
            return self

        if y is None:
            y = x

        y = _as_torch(y)  # type: ignore[has-type]
        self.data_max_ = torch.max(y, dim=0)[0].view_as(self.data_max_)
        return self

    def transform(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Transform the data using the fitted scaler into [0, max_].

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The data to transform.
        y : np.ndarray or torch.Tensor, optional
            The data to transform. If None, x is used.
            Default is None.

        Returns
        -------
        torch.Tensor
            The transformed data in the range [0, max_].
        """
        if y is None:
            y = x
        return self.forward(_as_torch(y))

    def inverse_transform(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Inverse transform the data from [0, max_] to the original scale using the fitted scaler.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The data to inverse transform.
        y : np.ndarray or torch.Tensor, optional
            The data to inverse transform. If None, x is used.
            Default is None.

        Returns
        -------
        torch.Tensor
            The inverse transformed data in the original scale.
        """
        if y is None:
            y = x

        y = _as_torch(y)  # type: ignore[has-type]
        return y * self.data_max_ / self.max_

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return torch.all(self.data_max_ != 1.0).item() or self.frozen_

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
