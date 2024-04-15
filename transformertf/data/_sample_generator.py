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
    "EncoderSample",
    "EncoderDecoderSample",
    "EncoderTargetSample",
    "EncoderDecoderTargetSample",
    "TransformerSampleGenerator",
    "TransformerPredictionSampleGenerator",
]

T = typing.TypeVar("T", np.ndarray, torch.Tensor)
U = typing.TypeVar("U")

log = logging.getLogger(__name__)


class TargetSample(TypedDict, typing.Generic[T]):
    target: T
    """ Target / ground truth data for the time series, size L. """


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
    def __getitem__(self, idx: int) -> U: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Sequence[U]: ...

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


class EncoderSample(TypedDict, typing.Generic[T]):
    encoder_input: T
    """ Source sequence to encoder. """
    encoder_mask: NotRequired[T]
    """ Source sequence mask to encoder. Typically should all be ones. """


class EncoderTargetSample(EncoderSample, TargetSample, typing.Generic[T]):
    x: int


class EncoderDecoderSample(EncoderSample, typing.Generic[T]):
    decoder_input: T
    """ Target sequence input to transformer. Typically should all be zeros. """
    decoder_mask: NotRequired[T]
    """ Target mask. Typically should all be ones. """


class EncoderDecoderTargetSample(
    EncoderDecoderSample, TargetSample, typing.Generic[T]
):
    pass


class TransformerSampleGenerator(
    SampleGenerator[EncoderDecoderTargetSample[T]]
):
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

    def _make_sample(self, idx: int) -> EncoderDecoderTargetSample[T]:  # type: ignore[override]
        idx = check_index(idx, len(self))

        sl = self._window_generator[idx]

        src_slice = slice(sl.start, sl.start + self._src_seq_len)
        tgt_slice = slice(sl.start + self._src_seq_len, sl.stop)

        src = concat(
            to_2dim(self._input_data[src_slice]),
            to_2dim(self._label_data[src_slice]),
            dim=-1,
        )  # [bs, seq_len, num_features]
        tgt = concat(
            to_2dim(self._input_data[tgt_slice]),
            to_2dim(zeros_like(self._label_data[tgt_slice])),
            dim=-1,
        )
        label = to_2dim(self._label_data[tgt_slice])

        return typing.cast(
            EncoderDecoderTargetSample[T],
            {
                "encoder_input": src,
                "encoder_mask": ones_like(src),
                "decoder_input": tgt,
                "decoder_mask": ones_like(tgt),
                "target": label,
            },
        )

    def __len__(self) -> int:
        return self._window_generator.num_samples


class TransformerPredictionSampleGenerator(
    SampleGenerator[EncoderDecoderSample[T]]
):
    def __init__(
        self,
        past_covariates: T,
        future_covariates: T,
        past_targets: T,
        context_length: int,
        prediction_length: int,
    ) -> None:
        super().__init__()

        self._num_points = len(future_covariates)

        if len(past_covariates) != len(past_targets):
            raise ValueError(
                "Past covariates and past target must have the same length: "
                f"({len(past_covariates)}) and ({len(past_targets)})"
            )
        if len(past_covariates) != context_length:
            raise ValueError(
                f"Past covariates must have length {context_length}"
            )

        self._context_length = context_length
        self._prediction_length = prediction_length
        self._total_context = len(future_covariates)

        self._window_generator = WindowGenerator(
            len(past_covariates) + len(future_covariates),
            context_length + prediction_length,
            stride=prediction_length,
            zero_pad=True,
        )

        future_covariates = zero_pad_(
            future_covariates, self._window_generator.real_data_len
        )
        self._covariates = to_2dim(
            concat(past_covariates, future_covariates, dim=0)
        )

        self._past_target = to_2dim(copy(past_targets))

    def add_target_context(self, future_target: T) -> None:
        """
        Add future target to the dataset to increase the context length.
        """
        if (
            len(future_target) + len(self._past_target)
            > self._total_context + self._context_length
        ):
            raise ValueError(
                "Future target length plus past target length must be "
                "less than or equal to the length of the future covariates "
                "since there can be no further predictions "
            )

        self._past_target = concat(
            self._past_target, to_2dim(future_target), dim=0
        )

    def _make_sample(self, idx: int) -> EncoderDecoderSample[T]:
        idx = check_index(idx, len(self))

        sl = self._window_generator[idx]
        ctxt_stop = sl.start + self._context_length

        src_slice = slice(sl.start, ctxt_stop)
        tgt_slice = slice(ctxt_stop, sl.stop)

        if ctxt_stop > len(self._past_target):
            raise IndexError(
                f"No context data available for index {idx}. "
                f"The context has ended at length {len(self._past_target)}, "
                f"but the index {idx} requires a length of {ctxt_stop}. "
                f"Add more target context using the `add_target_context` method."
            )

        src = concat(
            to_2dim(self._covariates[src_slice]),
            to_2dim(self._past_target[src_slice]),
            dim=-1,
        )
        tgt = concat(
            to_2dim(self._covariates[tgt_slice]),
            to_2dim(zeros_like(self._covariates[tgt_slice][..., 0])),
            dim=-1,
        )

        return typing.cast(
            EncoderDecoderSample[T],
            {
                "encoder_input": src,
                "encoder_mask": ones_like(src),
                "decoder_input": tgt,
                "decoder_mask": ones_like(tgt),
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
        return np.stack(arrs, axis=dim)  # pyright: ignore [reportGeneralTypeIssues]
    elif all(isinstance(arr, torch.Tensor) for arr in arrs):
        return torch.stack(arrs, dim=dim)  # pyright: ignore [reportGeneralTypeIssues]
    else:
        raise TypeError


def concat(*arrs: T, dim: int = 0) -> T:
    if all(isinstance(arr, np.ndarray) for arr in arrs):
        return np.concatenate(arrs, axis=dim)  # type: ignore[return-value]
    elif all(isinstance(arr, torch.Tensor) for arr in arrs):
        return torch.cat(arrs, dim=dim)  # type: ignore[return-value]
    else:
        raise TypeError


def to_2dim(arr: T) -> T:
    if arr.ndim == 0:
        return arr[None, None]
    elif arr.ndim == 1:
        return arr[..., None]
    else:
        return arr
