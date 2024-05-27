"""
This module contains the :class:`HysteresisDataset` and :class:`DataSource` classes,
which are used to load and preprocess data for training the hysteresis model.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

import enum
import logging

import numpy as np
import torch

from ..transform import BaseTransform

log = logging.getLogger(__name__)


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
