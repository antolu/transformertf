"""
This module contains functions for preprocessing data.

:author: Anton Lu (anton.lu@cern.ch)
"""
from __future__ import annotations

import logging
import math
import typing

import numpy as np
import pandas as pd

__all__ = ["WindowGenerator"]

log = logging.getLogger(__name__)

DATA_SOURCE = typing.TypeVar(
    "DATA_SOURCE", pd.Series, np.ndarray, covariant=True
)


class WindowGenerator:
    """
    Class to generate sliding windows for input and label data.

    The class takes both input and label data, but takes different window sizes
    for each. If the label window size is not specified, it is set to the same
    size as the input window size.

    The class does not consider the number of columns in the input and label
    data, therefore the returned data is not necessarily of the same shape.
    """

    def __init__(
        self,
        input_data: DATA_SOURCE,
        in_window_size: int,
        label_data: DATA_SOURCE | None = None,
        label_seq_len: int | None = None,
        stride: int = 1,
        zero_pad: bool = False,
    ):
        """
        :param input_data: Input data. Can be a pandas DataFrame, Series, or
            numpy array.
        :param in_window_size: Input window size.
        :param label_data: Label data. Can be a pandas DataFrame, Series, or
            numpy array.
        :param label_seq_len: Label window size. If None, the input window size
            is used.
        :param stride: Stride for the sliding window.
        :param zero_pad: If True, the input and label data are zero padded to
            fit the stride.
        """
        if label_seq_len is None:
            label_seq_len = in_window_size

        self._label_window_size = label_seq_len
        self._input_data = np.array(input_data)

        if label_data is None:
            self._label_data = None
        else:
            self._label_data = np.array(label_data)

            if len(self._input_data) != len(self._label_data):
                raise ValueError(
                    "Input and label data must have the same length: "
                    "({}) and ({})".format(
                        len(self._input_data), len(self._label_data)
                    )
                )

        if len(self._input_data) < in_window_size:
            raise ValueError(
                "Input data length ({}) must be greater than the input window "
                "size ({})".format(len(self._input_data), in_window_size)
            )

        self._in_window_size = in_window_size
        self._stride = stride

        if not zero_pad:
            self._num_samples = int(
                (len(self._input_data) - self._in_window_size) // self._stride
                + 1
            )
        else:
            self._num_samples = math.ceil(len(self._input_data) / stride)

        if zero_pad:
            input_data_padded = np.zeros(
                (self._num_samples * self._stride, *input_data.shape[1:])
            )
            input_data_padded[: len(self._input_data)] = self._input_data
            self._input_data = input_data_padded

            if label_data is not None:
                label_data_padded = np.zeros(
                    (self._num_samples * self._stride, *label_data.shape[1:])  # type: ignore[union-attr]  # noqa: E501
                )
                label_data_padded[: len(label_data)] = label_data
                self._label_data = label_data_padded

    @property
    def num_samples(self) -> int:
        """
        Number of samples that can be generated from the input data.
        :return: Number of samples.
        """
        return self._num_samples

    def __len__(self) -> int:
        return self.num_samples

    def calc_slice(self, idx: int, label: bool = False) -> slice:
        """
        Calculate the slice (window size) for the input or label data at a
        specific index. First performs an index check to ensure that the index
        is within bounds.

        :param idx: Index of the sample. If index is out of bounds an
            IndexError is raised.
        :param label: If True, the label window size is used. Otherwise, the
            input window size is used.
        :return: Slice object representing the window size.
        """
        # Check index
        if idx < 0:
            idx = idx + self.num_samples

        if idx >= self._num_samples:
            raise IndexError(f"Index {idx} is out of bounds")

        # Calculate slice
        start = idx * self._stride
        if label:
            stop = start + self._label_window_size
        else:
            stop = start + self._in_window_size

        return slice(start, stop)

    def get_sample(
        self, idx: int
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculate sliding window for input and label data.
        Return a pair of input and labeled data.
        The input and labeled data are not necessarily of the same length

        :param idx: index of the sample. If index is out of bounds an
            IndexError is raised.
        :return: Tuple of input and label data (x, y).
        """
        input_data = self._input_data[self.calc_slice(idx)]
        if self._label_data is None:
            return input_data
        else:
            return (
                input_data,
                self._label_data[self.calc_slice(idx, True)],
            )

    def __getitem__(
        self, idx: int
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self.get_sample(idx)
