"""
This module contains the :class:`HysteresisDataset` and :class:`DataSource` classes,
which are used to load and preprocess data for training the hysteresis model.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

import enum
import logging
import typing

import numpy as np
import pandas as pd
import torch

from ..transform import BaseTransform

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="AbstractTimeSeriesDataset")


log = logging.getLogger(__name__)

DATA_SOURCE: typing.TypeAlias = pd.Series | np.ndarray | torch.Tensor

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int": torch.int64,
    "long": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def get_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, str):
        return DTYPE_MAP[dtype]
    return dtype


class DataSetType(enum.Enum):
    TRAIN = "train"
    VAL_TEST = "validation_test"
    PREDICT = "predict"


class AbstractTimeSeriesDataset(torch.utils.data.Dataset):
    _input_data: list[torch.Tensor]
    _target_data: list[torch.Tensor] | list[None]
    _input_transform: dict[str, BaseTransform]
    _target_transform: BaseTransform | None
    _dataset_type: DataSetType

    @property
    def num_points(self) -> int:
        """
        The total number of points in the dataset.
        This is different from :meth:`__len__` that gives the number of
        samples in the dataset. This is the number of points in the original
        dataframes.

        :return: The number of points.
        """
        return int(np.sum([len(arr) for arr in self._input_data]))

    @property
    def input_transform(self) -> dict[str, BaseTransform]:
        return self._input_transform

    @property
    def target_transform(self) -> BaseTransform | None:
        return self._target_transform


def convert_data(
    data: DATA_SOURCE | list[DATA_SOURCE],
    dtype: torch.dtype | str = torch.float32,
) -> list[torch.Tensor]:
    source = data if isinstance(data, list) else [data]

    def to_torch(
        o: pd.Series | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(o, pd.Series):
            return torch.from_numpy(o.to_numpy())
        if isinstance(o, np.ndarray):
            return torch.from_numpy(o)
        if isinstance(o, torch.Tensor):
            return o
        msg = f"Unsupported type {type(o)} for data"
        raise TypeError(msg)

    dtype = DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    return [to_torch(o).to(dtype) for o in source]


def _check_index(idx: int, length: int) -> int:
    """
    Checks if an index for __getitem__ is valid.
    """
    if idx > length or idx < -length:
        msg = f"Index {idx} is out of bounds for dataset with " f" {length} samples."
        raise IndexError(msg)

    if idx < 0:
        idx += length

    return idx


def _check_label_data_length(
    input_data: list[torch.Tensor],
    target_data: list[torch.Tensor] | list[None],
) -> None:
    """
    This function checks the length of the label data sources
    and raises an error if they are not the same.
    This function should only be called when label data is
    present.
    """
    if len(input_data) != len(target_data):
        msg = "The number of input and target data sources must be the same."
        raise ValueError(msg)
    if not all(
        target is not None and len(input_) == len(target)
        for input_, target in zip(input_data, target_data, strict=False)
    ):
        msg = (
            "The number of samples in the input and target data "
            "sources must be the same."
        )
        raise ValueError(msg)
