from __future__ import annotations

import enum
import logging
import typing

import numpy as np
import pandas as pd
import sklearn.base
import torch
import torch.nn as nn
import tqdm

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="BaseTransform")

log = logging.getLogger(__name__)


__all__ = [
    "BaseTransform",
    "PolynomialTransform",
    "RunningNormalizer",
    "TransformCollection",
]


class TransformType(enum.Enum):
    X = "X"
    XY = "XY"
    COLLECTION = "COLLECTION"
    UNSPECIFIED = "UNSPECIFIED"


class BaseTransform(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, nn.Module
):
    _transform_type: TransformType = TransformType.UNSPECIFIED

    def __init__(self) -> None:
        sklearn.base.BaseEstimator.__init__(self)
        sklearn.base.TransformerMixin.__init__(self)
        nn.Module.__init__(self)

    @typing.overload
    def fit(self: SameType, x: torch.Tensor | np.ndarray) -> SameType:
        ...

    @typing.overload
    def fit(
        self: SameType,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> SameType:
        ...

    @typing.overload
    def fit(
        self: SameType,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> SameType:
        ...

    def fit(
        self: SameType,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> SameType:
        raise NotImplementedError

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Applies the inverse transformation on the input data.
        If y is None, then the inverse transform is applied to x.
        This is useful for when the target transform is dependent
        also on the input data.
        """
        raise NotImplementedError

    def __sklearn_is_fitted__(self) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return str(self)


class TransformCollection(BaseTransform):
    def __init__(
        self,
        transforms: list[BaseTransform],
        transform_type: TransformType = TransformType.X,
    ):
        """
        A collection of transforms that can be applied to the data.
        Depending on the transform type, the transforms are fitted/transformed
        to x or y, but dependent on x.

        This can be useful if the transformation on the target data is dependent
        on the input data, e.g. a polynomial transform.

        Parameters
        ----------
        transforms : list[BaseTransform]
            A list of transforms to apply to the data.
        transform_type : TransformType, optional
            The transform type of the collection, by default TransformType.X
            i.e. the transforms are fitted/transformed to x.
            If TransformType.XY, then the transforms are fitted/transformed to y.
            If there are transforms of both types, then the X-only transforms are
            fitted to the Y data.
        """
        super().__init__()
        self.transforms = transforms
        self._transform_type = transform_type

    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> TransformCollection:
        x_transformed = _as_torch(x)
        y_transformed = _as_torch(y) if y is not None else None

        for transform in self.transforms:
            if transform._transform_type == TransformType.X:
                if self._transform_type == TransformType.X:
                    x_transformed = transform.fit_transform(x_transformed)
                elif self._transform_type == TransformType.XY:
                    if y_transformed is None:
                        raise ValueError("Cannot fit Y when Y is None.")
                    y_transformed = transform.fit_transform(y_transformed)
            elif transform._transform_type == TransformType.XY:
                if self._transform_type == TransformType.X:
                    x_transformed = transform.fit_transform(x_transformed, y_transformed)
                elif self._transform_type == TransformType.XY:
                    y_transformed = transform.fit_transform(
                        x_transformed, y_transformed
                    )
            else:
                raise ValueError(
                    f"Invalid transform type: {transform._transform_type}"
                )
        return self

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        x_transformed = _as_torch(x)
        y_transformed = _as_torch(y) if y is not None else None

        for transform in self.transforms:
            if transform._transform_type == TransformType.X:
                if self._transform_type == TransformType.X:
                    x_transformed = transform.transform(x_transformed)
                elif self._transform_type == TransformType.XY:
                    if y_transformed is None:
                        raise ValueError("Cannot transform Y when Y is None.")
                    y_transformed = transform.transform(y_transformed)
                else:
                    raise ValueError(
                        f"Invalid transform type: {self._transform_type}"
                    )

            elif transform._transform_type == TransformType.XY:
                y_transformed = transform.transform(
                    x_transformed, y_transformed
                )

            else:
                raise ValueError(
                    f"Invalid transform type: {transform._transform_type}"
                )

        if y_transformed is None:
            return x_transformed

        return y_transformed

    def inverse_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        x_transformed = _as_torch(x)
        y_transformed = _as_torch(y) if y is not None else None

        for transform in reversed(self.transforms):
            if transform._transform_type == TransformType.X:
                if self._transform_type == TransformType.X:
                    x_transformed = transform.inverse_transform(x_transformed)
                elif self._transform_type == TransformType.XY:
                    if y_transformed is None:
                        raise ValueError("Cannot transform Y when Y is None.")
                    y_transformed = transform.inverse_transform(y_transformed)

            elif transform._transform_type == TransformType.XY:
                y_transformed = transform.inverse_transform(
                    x_transformed, y_transformed
                )

            else:
                raise ValueError(
                    f"Invalid transform type: {transform._transform_type}"
                )

        if y_transformed is None:
            return x_transformed

        return y_transformed

    # def state_dict(self) -> dict[str, typing.Any]:
    #     return {
    #         transform.__class__.__name__: transform.state_dict()
    #         for transform in self.transforms
    #     }

    # def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
    #     for transform in self.transforms:
    #         transform.load_state_dict(state_dict[transform.__class__.__name__])

    def __sklearn_is_fitted__(self) -> bool:
        return all(
            transform.__sklearn_is_fitted__() for transform in self.transforms
        )


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
        num_features : int
            The number of features in the input data.

        center : torch.Tensor or float, optional
            The center value for normalization. Default is 0.0.

        scale : torch.Tensor or float, optional
            The scale value for normalization. Default is 1.0.

        """
        BaseTransform.__init__(self)

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
    ) -> torch.Tensor:
        ...

    @typing.overload
    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
        return_scales: typing.Literal[True] = True,
        target_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"


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
