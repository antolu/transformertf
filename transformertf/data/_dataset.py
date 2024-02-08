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

from ._sample_generator import (EncoderDecoderSample, EncoderSample,
                                TimeSeriesSample, TimeSeriesSampleGenerator,
                                TransformerSampleGenerator)
from .transform import BaseTransform

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="AbstractTimeSeriesDataset")

__all__ = [
    "AbstractTimeSeriesDataset",
    "TimeSeriesDataset",
    "EncoderDataset",
    "EncoderDecoderDataset",
]


log = logging.getLogger(__name__)

DATA_SOURCE = typing.Union[pd.Series, np.ndarray, torch.Tensor]


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


class TimeSeriesDataset(AbstractTimeSeriesDataset):
    _input_data: list[torch.Tensor]
    _target_data: list[torch.Tensor] | list[None]

    def __init__(
        self,
        input_data: DATA_SOURCE | list[DATA_SOURCE],
        seq_len: int | None = None,
        target_data: DATA_SOURCE | list[DATA_SOURCE] | None = None,
        *,
        stride: int = 1,
        predict: bool = False,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        input_transform: dict[str, BaseTransform] | None = None,
        target_transform: BaseTransform | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        The dataset used to train the hysteresis model.

        Multiple dataframes can be provided, and the class will calculate
        samples from each dataframe on-demand in the __getitem__ method.
        This is done to save memory on larger datasets.
        The dataframes are never merged/concatenated into a single dataset
        as the gap between the timestamps in the dataframes represent
        a discontinuity in the data.

        The samples are created by creating "sliding windows" from the input
        and the target data.
        The input data is the input current, and the target is the magnetic flux
        (with or without its derivative.). The dimensions of the input and targets are
        (in_seq_len, 1) and (in_seq_len + out_seq_len, 2) respectively.
        This is concatenated into a single torch.Tensor of shape (batch, seq_len, 1 or 2) during training.

        Parameters
        ----------
        seq_len : int, optional
            The length of the windows to create for the samples. If none, the
            entire dataset is used.
        stride : int
            The offset between the start of each sliding window. Defaults to 1.
        predict : bool
            Whether the dataset is used for prediction or training. When predicting,
            the stride is ignored and set to the length of the window, i.e. the
            :attr:`seq_len` parameter.
        min_seq_len : int, optional
            The minimum length of the sliding window. This is used when
            :attr:`randomize_seq_len` is True.
        randomize_seq_len : bool
            Randomize the length of the sliding window. The length is chosen
            with uniform probability between :attr:`min_seq_len` and
            :attr:`seq_len`. This is used to train the model to handle
            sequences of different lengths.
        target_transform : BaseTransform, optional
            The transform that was used to transform the target data.
            This is used to inverse transform the target data.
            For the transforms for the input data, use the transforms
            encapsulated in the :class:`DataModuleBase` class.
        dtype : torch.dtype
            The data type of the torch.Tensors returned by the dataset in the
            __getitem__ method. Defaults to torch.float32.
        """
        super().__init__()

        if randomize_seq_len:
            if seq_len is None:
                raise ValueError(
                    "seq_len must be specified when randomize_seq_len is True."
                )
            if min_seq_len is None:
                raise ValueError(
                    "min_seq_len must be specified when randomize_seq_len is True."
                )
            if min_seq_len > seq_len:
                raise ValueError(
                    "min_seq_len must be less than or equal to seq_len."
                )

        self._input_data = convert_data(input_data, dtype=dtype)

        if target_data is not None:
            self._target_data = convert_data(target_data, dtype=dtype)
            _check_label_data_length(self._input_data, self._target_data)

            # if there is labeled data, it's either for training or validation
            if predict:
                self._dataset_type = DataSetType.VAL_TEST
            else:
                self._dataset_type = DataSetType.TRAIN

        else:  # no label predict
            if not predict:
                raise ValueError(
                    "Cannot use predict=False with no label data."
                )

            self._target_data = [None]
            self._dataset_type = DataSetType.PREDICT

            if len(self._input_data) > 1:
                raise ValueError(
                    f"Predicting requires exactly one input data source. "
                    f"Got {len(self._input_data)}"
                )

        if seq_len is None:
            if len(self._input_data) > 1:
                raise ValueError(
                    f"seq_len must be specified when using more than one input data source. "
                    f"Got {len(self._input_data)} input data sources."
                )
            else:
                seq_len = len(self._input_data[0])

        if predict:
            if stride != 1:
                log.warning("Stride is ignored when predicting.")
            stride = seq_len

        self._seq_len = seq_len
        self._min_seq_len = min_seq_len
        self._randomize_seq_len = randomize_seq_len
        self._stride = stride
        self._predict = predict

        self._input_transform = input_transform or {}
        self._target_transform = target_transform

        self._sample_gen = [
            TimeSeriesSampleGenerator(
                input_data=input_,
                window_size=seq_len,
                label_data=target,
                stride=stride,
                zero_pad=predict,
            )
            for input_, target in zip(self._input_data, self._target_data)
        ]

        # needed to determine which dataframe to get samples from
        self._cum_num_samples = np.cumsum(
            [len(gen) for gen in self._sample_gen]
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        input_columns: str | typing.Sequence[str],
        target_column: str | None = None,
        seq_len: int | None = None,
        stride: int = 1,
        predict: bool = False,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        input_transform: dict[str, BaseTransform] | None = None,
        target_transform: BaseTransform | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TimeSeriesDataset:
        if isinstance(input_columns, str):
            input_columns = [input_columns]

        input_data = dataframe[list(input_columns)].to_numpy()

        if target_column is None:
            target_data = None
        else:
            target_data = dataframe[list(target_column)].to_numpy()

        return cls(
            input_data=input_data,
            seq_len=seq_len,
            target_data=target_data,
            stride=stride,
            predict=predict,
            min_seq_len=min_seq_len,
            randomize_seq_len=randomize_seq_len,
            input_transform=input_transform,
            target_transform=target_transform,
            dtype=dtype,
        )

    def __getitem__(self, idx: int) -> TimeSeriesSample:
        """
        Get a single sample from the dataset.

        :param idx: The index of the sample to get.
        :return: A tuple of the input and target torch.Tensors.
        """
        if self._dataset_type in (DataSetType.TRAIN, DataSetType.VAL_TEST):
            return self._create_sample(idx)
        elif self._dataset_type == DataSetType.PREDICT:
            return self._get_prediction_input(idx)
        else:
            raise ValueError(f"Unknown dataset type {self._dataset_type}")

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self._cum_num_samples[-1]

    def __iter__(self) -> typing.Iterator[TimeSeriesSample]:
        for i in range(len(self)):
            yield self[i]

    def _create_sample(self, idx: int) -> TimeSeriesSample:
        """
        Create a single sample from the dataset. Used internally by the __getitem__ method.

        :param idx: The index of the sample to get.

        :return: A tuple of the input and target torch.Tensors.
        """
        idx = _check_index(idx, len(self))

        # find which df to get samples from
        df_idx = np.argmax(self._cum_num_samples > idx)

        shifted_idx = (
            idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx
        )

        sample = self._sample_gen[df_idx][shifted_idx]

        if self._randomize_seq_len:
            assert self._min_seq_len is not None
            random_len = np.random.randint(self._min_seq_len, self._seq_len)
            sample["input"][random_len:] = 0.0

            if "target" in sample:
                sample["target"][random_len:] = 0.0

        # add dimension for num features
        if sample["input"].ndim == 1:
            sample["input"] = sample["input"][..., None]
        if "target" in sample and sample["target"].ndim == 1:
            sample["target"] = sample["target"][..., None]

        return sample

    def _get_prediction_input(self, idx: int) -> TimeSeriesSample:
        """
        Get a single prediction input from the dataset.

        :param idx: The index of the sample to get.
        :return: A torch.Tensor.
        """
        idx = _check_index(idx, len(self))

        x = self._sample_gen[0][idx]

        if self._randomize_seq_len:
            assert self._min_seq_len is not None
            random_len = np.random.randint(self._min_seq_len, self._seq_len)
            x["input"][random_len:] = 0.0

        return x


class EncoderDataset(AbstractTimeSeriesDataset):
    def __init__(
        self,
        input_data: DATA_SOURCE | list[DATA_SOURCE],
        target_data: DATA_SOURCE | list[DATA_SOURCE],
        ctx_seq_len: int,
        tgt_seq_len: int,
        *,
        stride: int = 1,
        predict: bool = False,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        input_transform: dict[str, BaseTransform] | None = None,
        target_transform: BaseTransform | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Dataset to train a transformer

        Parameters
        ----------
        input_data
        target_data
        ctx_seq_len
        tgt_seq_len
        stride
        predict
        min_ctxt_seq_len
        min_tgt_seq_len
        randomize_seq_len
        target_transform
        dtype
        """
        super().__init__()

        if randomize_seq_len:
            if min_tgt_seq_len is None:
                raise ValueError(
                    "min_tgt_seq_len must be specified when "
                    "randomize_seq_len is True"
                )
            if min_ctxt_seq_len is None:
                raise ValueError(
                    "min_ctx_seq_len must be specified when "
                    "randomize_seq_len is True"
                )

        self._input_data = convert_data(input_data, dtype=dtype)
        self._target_data = convert_data(target_data, dtype=dtype)
        _check_label_data_length(self._input_data, self._target_data)

        self._dataset_type = (
            DataSetType.VAL_TEST if predict else DataSetType.TRAIN
        )

        if predict:
            if stride != 1:
                log.warning("Stride is ignored when predicting.")
            stride = tgt_seq_len
            if randomize_seq_len:
                # TODO: allow random seq len for validation purposes
                log.warning("randomize_seq_len is ignored when predicting.")

                randomize_seq_len = False

        self._ctxt_seq_len = ctx_seq_len
        self._tgt_seq_len = tgt_seq_len
        self._predict = predict
        self._stride = stride
        self._randomize_seq_len = randomize_seq_len

        self._input_transform = input_transform or {}
        self._target_transform = target_transform

        self._sample_gen = [
            TransformerSampleGenerator(
                input_data=input_,
                target_data=target_,
                src_seq_len=ctx_seq_len,
                tgt_seq_len=tgt_seq_len,
                zero_pad=predict,
                stride=stride,
            )
            for input_, target_ in zip(
                self._input_data,
                self._target_data,
            )
        ]

        self._cum_num_samples = np.cumsum(
            [len(gen) for gen in self._sample_gen]
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        input_columns: str | typing.Sequence[str],
        target_column: str,
        ctx_seq_len: int,
        tgt_seq_len: int,
        *,
        stride: int = 1,
        predict: bool = False,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        target_transform: BaseTransform | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> EncoderDecoderDataset:
        if isinstance(input_columns, str):
            input_columns = [input_columns]

        input_data = dataframe[list(input_columns)].to_numpy()
        target_data = dataframe[target_column].to_numpy()

        return cls(
            input_data=input_data,
            target_data=target_data,
            ctx_seq_len=ctx_seq_len,
            tgt_seq_len=tgt_seq_len,
            stride=stride,
            predict=predict,
            min_ctxt_seq_len=min_ctxt_seq_len,
            min_tgt_seq_len=min_tgt_seq_len,
            randomize_seq_len=randomize_seq_len,
            target_transform=target_transform,
            dtype=dtype,
        )

    def __getitem__(self, idx: int) -> EncoderSample:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx

        Returns
        -------
        EncoderSample
        """
        idx = _check_index(idx, len(self))

        # find which df to get samples from
        df_idx = np.argmax(self._cum_num_samples > idx)

        shifted_idx = (
            idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx
        )

        sample = self._sample_gen[df_idx][shifted_idx]

        if self._randomize_seq_len:
            assert self._min_ctxt_seq_len is not None
            random_len = np.random.randint(
                self._min_ctxt_seq_len, self._ctxt_seq_len
            )
            sample["encoder_input"][random_len:] = 0.0

            random_len = np.random.randint(
                self._min_tgt_seq_len, self._tgt_seq_len
            )
            sample["decoder_input"][random_len:] = 0.0

            if "target" in sample:
                sample["target"][random_len:] = 0.0

        # concatenate input and target data
        target_old = sample["encoder_input"][..., -1, None]

        target = torch.concat((target_old, sample["target"]), dim=0)
        encoder_input = torch.concat(
            (sample["encoder_input"], sample["decoder_input"]),
            dim=0,
        )

        sample = {
            "encoder_input": encoder_input,
            "encoder_mask": torch.ones_like(encoder_input),
            "target": target,
        }

        return sample

    def __len__(self) -> int:
        return self._cum_num_samples[-1]


class EncoderDecoderDataset(EncoderDataset):
    def __getitem__(self, idx: int) -> EncoderDecoderSample:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx

        Returns
        -------
        EncoderDecoderSample
        """
        idx = _check_index(idx, len(self))

        # find which df to get samples from
        df_idx = np.argmax(self._cum_num_samples > idx)

        shifted_idx = (
            idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx
        )

        sample = self._sample_gen[df_idx][shifted_idx]

        if self._randomize_seq_len:
            assert self._min_ctxt_seq_len is not None
            random_len = np.random.randint(
                self._min_ctxt_seq_len, self._ctxt_seq_len
            )
            sample["encoder_input"][random_len:] = 0.0

            random_len = np.random.randint(
                self._min_tgt_seq_len, self._tgt_seq_len
            )
            sample["decoder_input"][random_len:] = 0.0

            if "target" in sample:
                sample["target"][random_len:] = 0.0

        return sample


def convert_data(
    data: DATA_SOURCE | list[DATA_SOURCE], dtype: torch.dtype = torch.float32
) -> list[torch.Tensor]:
    source = data if isinstance(data, list) else [data]

    def to_torch(
        o: pd.Series | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(o, pd.Series):
            return torch.from_numpy(o.to_numpy())
        elif isinstance(o, np.ndarray):
            return torch.from_numpy(o)
        elif isinstance(o, torch.Tensor):
            return o
        else:
            raise TypeError(f"Unsupported type {type(o)} for data")

    dtype = DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    return [to_torch(o).to(dtype) for o in source]


def _check_index(idx: int, length: int) -> int:
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
        raise ValueError(
            "The number of input and target data sources must be the same."
        )
    if not all(
        [
            target is not None and len(input_) == len(target)
            for input_, target in zip(input_data, target_data)
        ]
    ):
        raise ValueError(
            "The number of samples in the input and target data "
            "sources must be the same."
        )
