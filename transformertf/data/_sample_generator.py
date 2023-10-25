"""
This module contains the TimeSeriesSampleGenerator and
TransformerSampleGenerator classes that are used to generate
samples for the time series dataset, using the WindowGenerator class.

:author: Anton Lu (anton.lu@cern.ch)
"""
from __future__ import annotations

import logging
import sys
import typing

import numpy as np
import torch

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import TypedDict, NotRequired

from ._window_generator import WindowGenerator

__all__ = [
    "TimeSeriesSample",
    "TimeSeriesSampleGenerator",
    "SampleGenerator",
    "TransformerSample",
    "TransformerSampleGenerator",
]

T = typing.TypeVar("T", np.ndarray, torch.Tensor)
U = typing.TypeVar("U")

log = logging.getLogger(__name__)


class TimeSeriesSample(TypedDict, typing.Generic[T]):
    input: T
    """ Input data for the time series, size L. """
    target: NotRequired[T]
    """ Target / ground truth data for the time series, size L."""
    initial_state: T
    """ Initial state of the time series. """


class SampleGenerator(typing.Sequence[U]):
    """
    Abstract base class for sample generators with common
    methods. The subclasses only need to implement the _make_sample
    and __len__ methods.
    """

    def _make_sample(self, idx: int) -> U:
        raise NotImplementedError

    @typing.overload
    def __getitem__(self, idx: int) -> U:
        ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Sequence[U]:
        ...

    def __getitem__(self, idx: int | slice) -> U | typing.Sequence[U]:
        if isinstance(idx, int):
            return self._make_sample(idx)
        else:
            return [
                self._make_sample(i)
                for i in range(
                    idx.start or 0, idx.stop or len(self), idx.step or 1
                )
            ]

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_samples={len(self)})"

    def __iter__(self) -> typing.Iterator[U]:
        for i in range(len(self)):
            yield self[i]


class TimeSeriesSampleGenerator(SampleGenerator[TimeSeriesSample[T]]):
    def __init__(
        self,
        input_data: T,
        window_size: int,
        label_data: T | None = None,
        stride: int = 1,
        zero_pad: bool = False,
    ):
        self._num_points = len(input_data)
        self._window_generator = WindowGenerator(
            self._num_points, window_size, stride, zero_pad
        )

        self._input_data: T = copy(input_data)

        if label_data is None:
            self._label_data = None
        else:
            self._label_data = copy(label_data)

            if len(self._input_data) != len(self._label_data):
                raise ValueError(
                    "Input and label data must have the same length: "
                    "({}) and ({})".format(
                        len(self._input_data), len(self._label_data)
                    )
                )

        if zero_pad:
            self._input_data = zero_pad_(
                self._input_data, self._window_generator.real_data_len
            )
            if self._label_data is not None:
                self._label_data = zero_pad_(
                    self._label_data, self._window_generator.real_data_len
                )

    def _make_sample(self, idx: int) -> TimeSeriesSample[T]:
        idx = check_index(idx, len(self))
        sl = self._window_generator[idx]

        input_data = self._input_data[sl]
        if input_data.ndim == 1:
            initial_x = input_data[0:1]
        else:
            initial_x = input_data[0]

        if self._label_data is None:
            return typing.cast(
                TimeSeriesSample[T],
                {
                    "input": input_data,
                    "initial_state": concat(initial_x, zeros_like(initial_x)),
                },
            )
        else:
            target_data = self._label_data[sl]
            if target_data.ndim == 1:
                initial_y = target_data[0:1]
            else:
                initial_y = target_data[0]

            return typing.cast(
                TimeSeriesSample[T],
                {
                    "input": input_data,
                    "target": target_data,
                    "initial_state": concat(initial_x, initial_y),
                },
            )

    def __len__(self) -> int:
        return self._window_generator.num_samples


class TransformerSample(TypedDict, typing.Generic[T]):
    encoder_input: T
    """ Source sequence to encoder. """
    encoder_mask: NotRequired[T]
    """ Source sequence mask to encoder. Typically should all be ones. """
    decoder_input: T
    """ Target sequence input to transformer. Typically should all be zeros. """
    decoder_mask: NotRequired[T]
    """ Target mask. Typically should all be ones. """

    target: T
    """ Target / ground truth sequence."""


class TransformerSampleGenerator(SampleGenerator[TransformerSample[T]]):
    _input_data: T
    _label_data: T

    def __init__(
        self,
        input_data: T,
        target_data: T,
        src_seq_len: int,
        tgt_seq_len: int,
        stride: int = 1,
        zero_pad: bool = False,
    ):
        self._num_points = len(input_data)
        self._window_generator = WindowGenerator(
            self._num_points, src_seq_len + tgt_seq_len, stride, zero_pad
        )
        self._src_seq_len = src_seq_len
        self._tgt_seq_len = tgt_seq_len

        self._input_data = copy(input_data)
        self._label_data = copy(target_data)

        if len(self._input_data) != len(self._label_data):
            raise ValueError(
                "Input and label data must have the same length: "
                "({}) and ({})".format(
                    len(self._input_data), len(self._label_data)
                )
            )

        if zero_pad:
            self._input_data = zero_pad_(
                self._input_data, self._window_generator.real_data_len
            )
            self._label_data = zero_pad_(
                self._label_data, self._window_generator.real_data_len
            )

    def _make_sample(self, idx: int) -> TransformerSample[T]:
        idx = check_index(idx, len(self))

        sl = self._window_generator[idx]

        src_slice = slice(sl.start, sl.start + self._src_seq_len)
        tgt_slice = slice(sl.start + self._src_seq_len, sl.stop)

        src = stack(self._input_data[src_slice], self._label_data[src_slice])
        tgt = stack(
            self._input_data[tgt_slice],
            zeros_like(self._input_data[tgt_slice]),
        )
        label = self._label_data[tgt_slice]

        if label.ndim == 1:
            label = label[..., None]

        return typing.cast(
            TransformerSample[T],
            {
                "encoder_input": src,
                "encoder_mask": ones_like(src),
                "decoder_input": tgt,
                "decoder_mask": ones_like(label),
                "target": label,
            },
        )

    def __len__(self) -> int:
        return self._window_generator.num_samples


def check_index(idx: int, length: int) -> int:
    """
    Checks if an index for __getitem__ is valid.
    """
    if idx > length or idx < -length:
        raise IndexError(
            f"Index {idx} is out of bounds for dataset with "
            f" {length} samples."
        )

    if idx < 0:
        idx += length

    return idx


def zeros_like(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    elif isinstance(arr, torch.Tensor):
        return torch.zeros_like(arr)
    else:
        raise TypeError(f"Unexpected type {type(arr)}")


def ones_like(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return np.ones_like(arr)
    elif isinstance(arr, torch.Tensor):
        return torch.ones_like(arr)
    else:
        raise TypeError


def zero_pad_(arr: T, length: int) -> T:
    if isinstance(arr, np.ndarray):
        zeros = np.zeros((length, *arr.shape[1:]), dtype=arr.dtype)
        zeros[: len(arr)] = arr
        return zeros
    elif isinstance(arr, torch.Tensor):
        zeros = torch.zeros(*(length, *arr.shape[1:]), dtype=arr.dtype)
        zeros[: len(arr)] = arr
        return zeros
    else:
        raise TypeError


def copy(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return arr.copy()
    elif isinstance(arr, torch.Tensor):
        return arr.clone()
    else:
        raise TypeError


def stack(*arrs: T, dim: int = -1) -> T:
    if all(isinstance(arr, np.ndarray) for arr in arrs):
        return np.stack(
            arrs, axis=dim
        )  # pyright: ignore [reportGeneralTypeIssues]
    elif all(isinstance(arr, torch.Tensor) for arr in arrs):
        return torch.stack(
            arrs, dim=dim
        )  # pyright: ignore [reportGeneralTypeIssues]
    else:
        raise TypeError


def concat(*arrs: T, dim: int = 0) -> T:
    if all(isinstance(arr, np.ndarray) for arr in arrs):
        return np.concatenate(arrs, axis=dim)  # type: ignore[return-value]
    elif all(isinstance(arr, torch.Tensor) for arr in arrs):
        return torch.cat(arrs, dim=dim)  # type: ignore[return-value]
    else:
        raise TypeError
