from __future__ import annotations

import typing

import numpy as np
import torch

from ._base import BaseTransform, TransformType
from ._utils import _as_torch, _view_as_y


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

    _transform_type = TransformType.X

    center_: torch.Tensor
    scale_: torch.Tensor
    n_samples_seen_: torch.Tensor

    def __init__(
        self,
        num_features_: int = 1,
        center_: torch.Tensor | float = 0.0,
        scale_: torch.Tensor | float = 1.0,
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

        Notes
        -----
        The arguments end with a `_` for scipy to detect them properly.
        """
        super().__init__()

        self.num_features_ = num_features_
        center_ = torch.zeros(num_features_, requires_grad=False) + center_
        scale_ = torch.zeros(num_features_, requires_grad=False) + scale_
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
        x = _as_torch(x)

        self._set_parameters(y_center=x, y_scale=x)
        return self

    @typing.overload
    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: None = None,
        return_scales: typing.Literal[False] = False,
        target_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @typing.overload
    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
        return_scales: typing.Literal[True] = True,
        target_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
        return_scales: bool = False,
        target_scale: torch.Tensor | None = None,
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
        new_mean = torch.mean(y_center, dim=dim, keepdim=True)
        new_scale = torch.std(y_scale, dim=dim, keepdim=True) + eps

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
