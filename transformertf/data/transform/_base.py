from __future__ import annotations

import enum
import logging
import typing

import numpy as np
import sklearn.base
import torch
from torch import nn

from ._utils import _as_torch

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="BaseTransform")

log = logging.getLogger(__name__)


__all__ = [
    "BaseTransform",
    "TransformCollection",
]


class BaseTransform(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, nn.Module
):
    class TransformType(enum.Enum):
        X = "X"
        XY = "XY"
        COLLECTION = "COLLECTION"
        UNSPECIFIED = "UNSPECIFIED"

    _transform_type: TransformType = TransformType.UNSPECIFIED

    def __init__(self) -> None:
        sklearn.base.BaseEstimator.__init__(self)
        sklearn.base.TransformerMixin.__init__(self)
        nn.Module.__init__(self)

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        """Ensure that the transform type is specified by the developer."""
        if cls._transform_type == cls.TransformType.UNSPECIFIED:
            msg = f"Transform type not specified for {cls.__name__}."
            raise ValueError(msg)

    @typing.overload
    def fit(self: SameType, x: torch.Tensor | np.ndarray) -> SameType: ...

    @typing.overload
    def fit(
        self: SameType,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> SameType: ...

    @typing.overload
    def fit(
        self: SameType,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> SameType: ...

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

    def fit_transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.fit(x, y).transform(x, y)

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

    @property
    def transform_type(self) -> TransformType:
        return self._transform_type

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return str(self)


class TransformCollection(BaseTransform, typing.Sequence[BaseTransform]):
    _transform_type = BaseTransform.TransformType.COLLECTION

    def __init__(
        self,
        *transforms: BaseTransform | typing.Sequence[BaseTransform],
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
        """
        super().__init__()
        transforms_exp = []
        for transform in transforms:
            if isinstance(transform, BaseTransform):
                transforms_exp.append(transform)
            elif isinstance(transform, typing.Sequence):
                transforms_exp.extend(list(transform))

        self.transforms = torch.nn.ModuleList(transforms_exp)
        transform_type = (
            self.TransformType.X
            if all(
                transform._transform_type == self.TransformType.X  # noqa: SLF001
                for transform in self.transforms
            )
            else self.TransformType.XY
        )
        self._transform_type = transform_type

    def fit(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> TransformCollection:
        x_transformed = _as_torch(x)
        y_transformed = _as_torch(y) if y is not None else None

        for transform in self.transforms:
            if self._transform_type == self.TransformType.X:
                if transform._transform_type == self.TransformType.X:  # noqa: SLF001
                    x_transformed = transform.fit_transform(x_transformed)
                elif transform._transform_type == self.TransformType.XY:  # noqa: SLF001
                    if y_transformed is None:
                        msg = "Cannot fit Y when Y is None."
                        raise ValueError(msg)
                    y_transformed = transform.fit_transform(y_transformed)
            elif self._transform_type == self.TransformType.XY:
                if transform._transform_type == self.TransformType.X:  # noqa: SLF001
                    y_transformed = transform.fit_transform(y_transformed)
                elif transform._transform_type == self.TransformType.XY:  # noqa: SLF001
                    y_transformed = transform.fit_transform(
                        x_transformed, y_transformed
                    )
            else:
                msg = f"Invalid transform type: {transform._transform_type}"  # noqa: SLF001
                raise ValueError(msg)
        return self

    def transform(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        x_transformed = _as_torch(x)
        y_transformed = _as_torch(y) if y is not None else None

        for transform in self.transforms:
            if self._transform_type == self.TransformType.X:
                if transform._transform_type == self.TransformType.X:  # noqa: SLF001
                    x_transformed = transform.transform(x_transformed)
                elif transform._transform_type == self.TransformType.XY:  # noqa: SLF001
                    if y_transformed is None:
                        msg = "Cannot transform Y when Y is None."
                        raise ValueError(msg)
                    y_transformed = transform.transform(y_transformed)
                else:
                    msg = f"Invalid transform type: {self._transform_type}"
                    raise ValueError(msg)

            elif self._transform_type == self.TransformType.XY:
                if transform._transform_type == self.TransformType.X:  # noqa: SLF001
                    y_transformed = transform.transform(y_transformed)
                elif transform._transform_type == self.TransformType.XY:  # noqa: SLF001
                    y_transformed = transform.transform(x_transformed, y_transformed)

            else:
                msg = f"Invalid transform type: {transform._transform_type}"  # noqa: SLF001
                raise ValueError(msg)

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

        if y is None and self._transform_type == self.TransformType.XY:
            msg = "Cannot transform Y when Y is None."
            raise ValueError(msg)

        for transform in reversed(self.transforms):
            if self._transform_type == self.TransformType.X:
                if transform._transform_type == self.TransformType.X:  # noqa: SLF001
                    x_transformed = transform.inverse_transform(x_transformed)
                elif transform._transform_type == self.TransformType.XY:  # noqa: SLF001
                    if y_transformed is None:
                        msg = "Cannot transform Y when Y is None."
                        raise ValueError(msg)
                    y_transformed = transform.inverse_transform(y_transformed)

            elif self._transform_type == self.TransformType.XY:
                if transform._transform_type == self.TransformType.X:  # noqa: SLF001
                    y_transformed = transform.inverse_transform(y_transformed)
                elif transform._transform_type == self.TransformType.XY:  # noqa: SLF001
                    y_transformed = transform.inverse_transform(
                        x_transformed, y_transformed
                    )

            else:
                msg = f"Invalid transform type: {transform._transform_type}"  # noqa: SLF001
                raise ValueError(msg)

        if y_transformed is None:
            return x_transformed

        return y_transformed

    @typing.overload
    def __getitem__(self, item: int) -> BaseTransform: ...

    @typing.overload
    def __getitem__(self, item: slice) -> TransformCollection: ...

    def __getitem__(self, item: int | slice) -> BaseTransform | TransformCollection:
        if isinstance(item, int):
            return self.transforms[item]
        return TransformCollection(*self.transforms[item])

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> typing.Iterator[BaseTransform]:
        return iter(self.transforms)

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        return all(transform.__sklearn_is_fitted__() for transform in self.transforms)
