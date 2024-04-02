from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import torch

from .._sample_generator import TimeSeriesSample, TimeSeriesSampleGenerator
from ..transform import BaseTransform
from ._base import (
    DATA_SOURCE,
    AbstractTimeSeriesDataset,
    DataSetType,
    _check_index,
    _check_label_data_length,
    convert_data,
)

log = logging.getLogger(__name__)


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
