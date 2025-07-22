"""
This module contains the TimeSeriesSampleGenerator and
TransformerSampleGenerator classes that are used to generate
samples for the time series dataset, using the WindowGenerator class.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

import logging
import typing
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
import torch

from ._window_generator import WindowGenerator

__all__ = [
    "EncoderDecoderSample",
    "EncoderDecoderTargetSample",
    "SampleGenerator",
    "TimeSeriesSample",
    "TimeSeriesSampleGenerator",
    "TransformerPredictionSampleGenerator",
    "TransformerSampleGenerator",
]

T = typing.TypeVar("T", np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)
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

    def __getitem__(self, idx: int | np.integer | slice) -> U | typing.Sequence[U]:
        if isinstance(idx, int | np.integer):
            return self._make_sample(int(idx))
        return [
            self._make_sample(i)
            for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)
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
        *,
        zero_pad: bool = False,
    ):
        self._num_points = len(input_data)
        self._window_generator = WindowGenerator(
            self._num_points, window_size, stride, zero_pad=zero_pad
        )

        self._input_data: T = copy(input_data)

        if label_data is None:
            self._label_data = None
        else:
            self._label_data = copy(label_data)

            if len(self._input_data) != len(self._label_data):
                msg = (
                    "Input and label data must have the same length: "
                    f"({len(self._input_data)}) and ({len(self._label_data)})"
                )
                raise ValueError(msg)

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

        if self._label_data is None:
            return typing.cast(
                TimeSeriesSample[T],
                {
                    "input": input_data,
                },
            )
        target_data = self._label_data[sl]

        return typing.cast(
            TimeSeriesSample[T],
            {
                "input": input_data,
                "target": target_data,
            },
        )

    def __len__(self) -> int:
        return self._window_generator.num_samples


class EncoderDecoderSample(TypedDict, typing.Generic[T]):
    encoder_input: T
    """ Source sequence to encoder. """
    encoder_mask: NotRequired[T]
    """ Source sequence mask to encoder. Typically should all be ones. """
    encoder_lengths: NotRequired[T]
    decoder_input: T
    """ Target sequence input to transformer. Typically should all be zeros. """
    decoder_mask: NotRequired[T]
    """ Target mask. Typically should all be ones. """
    decoder_lengths: NotRequired[T]


class EncoderDecoderTargetSample(EncoderDecoderSample, TargetSample, typing.Generic[T]):
    target_mask: NotRequired[T]


class TransformerSampleGenerator(SampleGenerator[EncoderDecoderTargetSample[T]]):
    _input_data: T
    _label_data: T

    def __init__(
        self,
        input_data: T,
        target_data: T,
        src_seq_len: int,
        tgt_seq_len: int,
        known_past_data: T | None = None,
        stride: int = 1,
        *,
        add_target_to_past: bool = True,
        zero_pad: bool = False,
    ):
        self._zero_pad = zero_pad
        self._add_target_to_past = add_target_to_past
        self._num_points = len(input_data)
        self._window_generator = WindowGenerator(
            self._num_points, src_seq_len + tgt_seq_len, stride, zero_pad=zero_pad
        )
        self._src_seq_len = src_seq_len
        self._tgt_seq_len = tgt_seq_len

        self._input_data = copy(input_data)
        self._label_data = copy(target_data)
        self._known_past_data: T | None = (
            copy(known_past_data) if known_past_data is not None else None
        )

        if len(self._input_data) != len(self._label_data):
            msg = (
                "Input and label data must have the same length: "
                f"({len(self._input_data)}) and ({len(self._label_data)})"
            )
            raise ValueError(msg)
        if self._known_past_data is not None and len(self._input_data) != len(
            self._known_past_data
        ):
            msg = (
                "Input and known past data must have the same length: "
                f"({len(self._input_data)}) and ({len(self._known_past_data)})"
            )
            raise ValueError(msg)

        if zero_pad:
            self._input_data = zero_pad_(
                self._input_data, self._window_generator.real_data_len
            )
            self._label_data = zero_pad_(
                self._label_data, self._window_generator.real_data_len
            )
            self._known_past_data = (
                zero_pad_(self._known_past_data, self._window_generator.real_data_len)
                if self._known_past_data is not None
                else None
            )

    def _make_sample(self, idx: int) -> EncoderDecoderTargetSample[T]:  # type: ignore[override]
        idx = check_index(idx, len(self))

        sl = self._window_generator[idx]

        src_slice = slice(sl.start, sl.start + self._src_seq_len)
        tgt_slice = slice(sl.start + self._src_seq_len, sl.stop)

        enc_in_l = [to_2dim(self._input_data[src_slice].copy())]
        if self._known_past_data is not None:
            enc_in_l.append(to_2dim(self._known_past_data[src_slice].copy()))
        if self._add_target_to_past:
            enc_in_l.append(to_2dim(self._label_data[src_slice].copy()))

        dec_in_l = [to_2dim(self._input_data[tgt_slice].copy())]
        if self._known_past_data is not None:
            dec_in_l.append(to_2dim(zeros_like(self._known_past_data[tgt_slice])))
        if self._add_target_to_past:
            # add dummy target data to decoder input
            # to ensure that the decoder input has the same shape as the encoder input
            dec_in_l.append(to_2dim(zeros_like(self._label_data[tgt_slice])))

        enc_in = concat(*enc_in_l, dim=-1)
        dec_in = concat(*dec_in_l, dim=-1)
        label = to_2dim(self._label_data[tgt_slice])

        decoder_mask = ones_like(dec_in)
        target_mask = ones_like(label)

        numel_pad = len(self._input_data) - self._num_points
        if idx == len(self) - 1 and self._zero_pad and numel_pad > 0:
            if not isinstance(decoder_mask, pd.DataFrame):
                decoder_mask[..., -numel_pad:] = 0.0
                target_mask[..., -numel_pad:] = 0.0
            else:
                decoder_mask.iloc[-numel_pad:] = 0.0
                target_mask.iloc[-numel_pad:] = 0.0

        if isinstance(enc_in, pd.DataFrame):
            enc_in = enc_in.reset_index(drop=True)
            dec_in = dec_in.reset_index(drop=True)
            label = label.reset_index(drop=True)
            decoder_mask = decoder_mask.reset_index(drop=True)
            target_mask = target_mask.reset_index(drop=True)

        return typing.cast(
            EncoderDecoderTargetSample[T],
            {
                "encoder_input": enc_in,
                "encoder_mask": ones_like(enc_in),
                "decoder_input": dec_in,
                "decoder_mask": decoder_mask,
                "target": label,
                "target_mask": target_mask,
            },
        )

    def __len__(self) -> int:
        return self._window_generator.num_samples


class TransformerPredictionSampleGenerator(SampleGenerator[EncoderDecoderSample[T]]):
    def __init__(
        self,
        past_covariates: T,
        future_covariates: T,
        past_targets: T,
        context_length: int,
        prediction_length: int,
        known_past_covariates: T | None = None,
    ) -> None:
        super().__init__()

        self._num_points = len(future_covariates)

        if len(past_covariates) != len(past_targets):
            msg = (
                "Past covariates and past target must have the same length: "
                f"({len(past_covariates)}) and ({len(past_targets)})"
            )
            raise ValueError(msg)
        if len(past_covariates) != context_length:
            msg = f"Past covariates must have length {context_length}"
            raise ValueError(msg)

        if (
            known_past_covariates is not None
            and len(known_past_covariates) != context_length
        ):
            msg = f"Known past covariates must have length {context_length}"
            raise ValueError(msg)

        self._context_length = context_length
        self._prediction_length = prediction_length
        self._total_context = len(future_covariates)

        wg_len = max(
            len(past_covariates) + len(future_covariates),
            context_length + prediction_length,
        )
        self._window_generator = WindowGenerator(
            wg_len,
            context_length + prediction_length,
            stride=prediction_length,
            zero_pad=True,
        )

        future_covariates = zero_pad_(
            future_covariates, self._window_generator.real_data_len
        )
        self._covariates: T = to_2dim(concat(past_covariates, future_covariates, dim=0))

        self._past_target: T = to_2dim(copy(past_targets))
        self._known_past_covariates: T | None = (
            to_2dim(copy(known_past_covariates))
            if known_past_covariates is not None
            else None
        )

        if isinstance(self._covariates, pd.DataFrame):
            self._covariates = self._covariates.reset_index(drop=True)
        if isinstance(self._past_target, pd.DataFrame):
            self._past_target = self._past_target.reset_index(drop=True)
        if self._known_past_covariates is not None and isinstance(
            self._known_past_covariates, pd.DataFrame
        ):
            self._known_past_covariates = self._known_past_covariates.reset_index(
                drop=True
            )

    def add_target_context(self, future_target: T) -> None:
        """
        Add future target to the dataset to increase the context length.
        """
        if (
            len(future_target) + len(self._past_target)
            > self._total_context + self._context_length
        ):
            msg = (
                "Future target length plus past target length must be "
                "less than or equal to the length of the future covariates "
                "since there can be no further predictions "
            )
            raise ValueError(msg)

        self._past_target = concat(self._past_target, to_2dim(future_target), dim=0)
        if isinstance(self._past_target, pd.DataFrame):
            self._past_target = self._past_target.reset_index(drop=True)

    def add_known_past_context(self, known_past_covariates: T) -> None:
        """
        Add known past covariates to the dataset to increase the context length.
        """
        if self._known_past_covariates is None:
            msg = "No known past covariates were provided during initialization"
            raise ValueError(msg)

        if (
            len(known_past_covariates) + len(self._known_past_covariates)
            > self._total_context + self._context_length
        ):
            msg = (
                "Known past covariates length plus past covariates length "
                "must be less than or equal to the context length"
            )
            raise ValueError(msg)

        self._known_past_covariates = concat(
            self._known_past_covariates, to_2dim(known_past_covariates), dim=0
        )
        if isinstance(self._known_past_covariates, pd.DataFrame):
            self._known_past_covariates = self._known_past_covariates.reset_index(
                drop=True
            )

    def _make_sample(self, idx: int) -> EncoderDecoderSample[T]:
        idx = check_index(idx, len(self))

        sl = self._window_generator[idx]
        ctxt_stop = sl.start + self._context_length

        src_slice = slice(sl.start, ctxt_stop)
        tgt_slice = slice(ctxt_stop, sl.stop)

        if ctxt_stop > len(self._past_target):
            msg = (
                f"No context data available for index {idx}. "
                f"The context has ended at length {len(self._past_target)}, "
                f"but the index {idx} requires a length of {ctxt_stop}. "
                f"Add more target context using the `add_target_context` method."
            )
            raise IndexError(msg)
        if self._known_past_covariates is not None and ctxt_stop > len(
            self._known_past_covariates
        ):
            msg = (
                f"No known past context data available for index {idx}. "
                f"The context has ended at length {len(self._known_past_covariates)}, "
                f"but the index {idx} requires a length of {ctxt_stop}. "
                f"Add more known past context using the `add_known_past_context` method."
            )
            raise IndexError(msg)

        src_l = [to_2dim(self._covariates[src_slice]).copy()]
        if self._known_past_covariates is not None:
            src_l.append(to_2dim(self._known_past_covariates[src_slice]).copy())
        src_l.append(to_2dim(self._past_target[src_slice]).copy())

        tgt_l = [to_2dim(self._covariates[tgt_slice]).copy()]
        if isinstance(tgt_l[0], pd.DataFrame):
            tgt_l[0] = tgt_l[0].reset_index(drop=True)
            tgt_slice = slice(
                0, tgt_slice.stop - tgt_slice.start
            )  # hack to get the indices
        if self._known_past_covariates is not None:
            tgt_l.append(
                to_2dim(
                    zeros(self._known_past_covariates[tgt_slice], length=tgt_slice.stop)
                )
            )
        tgt_l.append(
            to_2dim(zeros(self._past_target[tgt_slice], length=tgt_slice.stop))
        )

        src = concat(*src_l, dim=-1)
        tgt = concat(*tgt_l, dim=-1)

        encoder_mask = ones_like(src)
        decoder_mask = ones_like(tgt)

        numel_pad = (
            self._window_generator.real_data_len
            - self._context_length
            - self._num_points
        )
        if idx == len(self) - 1 and numel_pad > 0:
            if not isinstance(decoder_mask, pd.DataFrame):
                decoder_mask[..., -numel_pad:] = 0.0
            else:
                decoder_mask.iloc[-numel_pad:] = 0.0

        if isinstance(src, pd.DataFrame):
            src = src.reset_index(drop=True)
            tgt = tgt.reset_index(drop=True)
            decoder_mask = decoder_mask.reset_index(drop=True)

        encoder_lengths: T
        if isinstance(src, pd.DataFrame):
            encoder_lengths = pd.DataFrame({"encoder_lengths": [self._context_length]})  # type: ignore[assignment]
        elif isinstance(src, torch.Tensor):
            encoder_lengths = torch.tensor([self._context_length])  # type: ignore[assignment]
        else:
            encoder_lengths = np.array([self._context_length])  # type: ignore[assignment]

        decoder_lengths: T
        if isinstance(tgt, pd.DataFrame):
            decoder_lengths = pd.DataFrame({  # type: ignore[assignment]
                "decoder_lengths": [self._prediction_length - numel_pad]
            })
        elif isinstance(tgt, torch.Tensor):
            decoder_lengths = torch.tensor(  # type: ignore[assignment]
                [self._prediction_length - numel_pad], dtype=torch.float32
            )
        else:
            decoder_lengths = np.array([self._prediction_length - numel_pad])  # type: ignore[assignment]

        return typing.cast(
            EncoderDecoderSample[T],
            {
                "encoder_input": src,
                "encoder_mask": encoder_mask,
                "encoder_lengths": encoder_lengths,
                "decoder_input": tgt,
                "decoder_mask": decoder_mask,
                "decoder_lengths": decoder_lengths,
            },
        )

    def __len__(self) -> int:
        return self._window_generator.num_samples


def check_index(idx: int, length: int) -> int:
    """
    Checks if an index for __getitem__ is valid.
    """
    if idx > length or idx < -length:
        msg = f"Index {idx} is out of bounds for dataset with  {length} samples."
        raise IndexError(msg)

    if idx < 0:
        idx += length

    return idx


EXC_MSG = (
    "Unexpected type {}, expected np.ndarray, torch.Tensor, pd.DataFrame or pd.Series"
)


def zeros(like: T, length: int) -> T:
    if isinstance(like, np.ndarray):
        return np.zeros((length, *like.shape[1:]), dtype=like.dtype)
    if isinstance(like, torch.Tensor):
        return torch.zeros(*(length, *like.shape[1:]), dtype=like.dtype)
    if isinstance(like, pd.DataFrame):
        return pd.DataFrame(index=range(length), columns=like.columns, data=0.0)
    if isinstance(like, pd.Series):
        return pd.Series(index=range(length), data=0.0, dtype=like.dtype)
    msg = EXC_MSG.format(type(like))
    raise TypeError(msg)


def zeros_like(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    if isinstance(arr, torch.Tensor):
        return torch.zeros_like(arr)
    if isinstance(arr, pd.DataFrame):
        return pd.DataFrame(index=arr.index, columns=arr.columns, data=0.0, dtype=float)
    if isinstance(arr, pd.Series):
        return pd.Series(index=arr.index, data=0)
    msg = EXC_MSG.format(type(arr))
    raise TypeError(msg)


def ones_like(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return np.ones_like(arr)
    if isinstance(arr, torch.Tensor):
        return torch.ones_like(arr)
    if isinstance(arr, pd.DataFrame):
        return pd.DataFrame(index=arr.index, columns=arr.columns, data=1.0, dtype=float)
    if isinstance(arr, pd.Series):
        return pd.Series(index=arr.index, data=1)
    msg = EXC_MSG.format(type(arr))
    raise TypeError(msg)


def zero_pad_(arr: T, length: int) -> T:
    zeros: T
    if isinstance(arr, np.ndarray):
        zeros = np.zeros((length, *arr.shape[1:]), dtype=arr.dtype)
        zeros[: len(arr)] = arr
        return zeros
    if isinstance(arr, torch.Tensor):
        zeros = torch.zeros(*(length, *arr.shape[1:]), dtype=arr.dtype)
        zeros[: len(arr)] = arr
        return zeros
    if isinstance(arr, pd.DataFrame):
        zeros = pd.concat(
            [
                arr,
                pd.DataFrame(
                    index=range(len(arr), length),
                    columns=arr.columns,
                    data=0.0,
                ),
            ],
            axis=0,
        )
        return zeros
    if isinstance(arr, pd.Series):
        zeros = pd.Series(index=range(length), data=0.0, dtype=arr.dtype)
        zeros.iloc[: len(arr)] = arr
        return zeros
    msg = EXC_MSG.format(type(arr))
    raise TypeError(msg)


def copy(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return arr.copy()
    if isinstance(arr, torch.Tensor):
        return arr.clone()
    if isinstance(arr, pd.DataFrame | pd.Series):
        return arr.copy()
    msg = EXC_MSG.format(type(arr))
    raise TypeError(msg)


def stack(*arrs: T, dim: int = -1) -> T:
    if all(isinstance(arr, np.ndarray) for arr in arrs):
        return np.stack(arrs, axis=dim)  # type: ignore[return-value]
    if all(isinstance(arr, torch.Tensor) for arr in arrs):
        return torch.stack(arrs, dim=dim)  # pyright: ignore [reportGeneralTypeIssues]
    if all(isinstance(arr, pd.DataFrame) for arr in arrs) or all(
        isinstance(arr, pd.Series) for arr in arrs
    ):
        return pd.concat(arrs, axis=dim)  # type: ignore[return-value,call-overload]
    (
        "All arrays must be of the same type, "
        "either np.ndarray, torch.Tensor, pd.DataFrame or pd.Series. Got {}"
    ).format(", ".join(type(arr).__name__ for arr in arrs))
    raise TypeError


def concat(*arrs: T, dim: int = 0) -> T:
    if all(isinstance(arr, np.ndarray) for arr in arrs):
        return np.concatenate(arrs, axis=dim)  # type: ignore[return-value]
    if all(isinstance(arr, torch.Tensor) for arr in arrs):
        return torch.cat(arrs, dim=dim)  # type: ignore[return-value]
    if all(isinstance(arr, pd.DataFrame | pd.Series) for arr in arrs):
        if dim == -1:
            dim = 1
        return pd.concat(arrs, axis=dim)  # type: ignore[return-value,call-overload]
    msg = (
        "All arrays must be of the same type, "
        "either np.ndarray, torch.Tensor, pd.DataFrame or pd.Series. "
        "Got {}".format(", ".join(type(arr).__name__ for arr in arrs))
    )

    raise TypeError(msg)


def to_2dim(arr: T) -> T:
    if isinstance(arr, pd.DataFrame | pd.Series):
        return arr
    if arr.ndim == 0:
        return arr[None, None]
    if arr.ndim == 1:
        return arr[..., None]
    if not hasattr(arr, "ndim"):
        msg = f"Object {arr} of type {type(arr)} does not have an ndim attribute."
        raise TypeError(msg)
    return arr
