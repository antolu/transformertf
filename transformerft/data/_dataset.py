"""
This module contains the :class:`HysteresisDataset` and :class:`DataSource` classes,
which are used to load and preprocess data for training the hysteresis model.

:author: Anton Lu (anton.lu@cern.ch)
"""
from __future__ import annotations

import enum
import logging
import sys
import typing
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import TypedDict, NotRequired

from ..modules import RunningNormalizer
from ._window_generator import WindowGenerator

__all__ = ["TimeSeriesDataset", "TimeSeriesSample"]


log = logging.getLogger(__name__)

TimeSeriesSample = TypedDict(
    "TimeSeriesSample",
    {"input": torch.Tensor, "target": NotRequired[torch.Tensor]},
)
DATA_SOURCE = typing.Union[pd.Series, np.ndarray, torch.Tensor]


class DataSetType(enum.Enum):
    TRAIN = "train"
    VAL_TEST = "validation_test"
    PREDICT = "predict"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        input_data: DATA_SOURCE | list[DATA_SOURCE],
        seq_len: int,
        target_data: DATA_SOURCE | list[DATA_SOURCE] | None = None,
        stride: int = 1,
        predict: bool = False,
        input_normalizer: RunningNormalizer | None = None,
        target_normalizer: RunningNormalizer | None = None,
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

        :param seq_len: The length of the windows into the current curve.
        :param stride: The offset between windows into the curve.
        :param zero_pad: Whether to zero-pad the input data to the length of the window
            if the input data is shorter than the window length.
        """
        super().__init__()

        if predict:
            if stride != 1:
                warnings.warn("Stride is ignored when predicting.")
            stride = seq_len

        self._seq_len = seq_len
        self._stride = stride
        self._predict = predict

        def convert_data(
            data: DATA_SOURCE | list[DATA_SOURCE],
        ) -> list[np.ndarray]:
            source = data if isinstance(data, list) else [data]

            def to_numpy(
                o: pd.Series | np.ndarray | torch.Tensor,
            ) -> np.ndarray:
                if isinstance(o, pd.Series):
                    return o.to_numpy()
                elif isinstance(o, np.ndarray):
                    return o
                elif isinstance(o, torch.Tensor):
                    return o.numpy()
                else:
                    raise TypeError(f"Unsupported type {type(o)} for data")

            return [to_numpy(o) for o in source]

        self._input_data = convert_data(input_data)

        self._target_data: typing.Sequence[np.ndarray | None]
        if target_data is not None:
            self._target_data = convert_data(target_data)

            if len(self._input_data) != len(self._target_data):
                raise ValueError(
                    "The number of input and target data sources must be the same."
                )
            if predict:
                self._dataset_type = DataSetType.VAL_TEST
            else:
                self._dataset_type = DataSetType.TRAIN
        else:
            self._target_data = [None] * len(self._input_data)
            self._dataset_type = DataSetType.PREDICT

            if predict and len(self._input_data) != 1:
                raise ValueError(
                    f"Predicting requires exactly one input data source. "
                    f"Got {len(self._input_data)}"
                )

        self._window_gen = [
            WindowGenerator(
                input_data=input_,
                in_window_size=seq_len,
                label_data=target,
                label_seq_len=seq_len,
                stride=stride,
                zero_pad=predict,
            )
            for input_, target in zip(self._input_data, self._target_data)
        ]

        self._cum_num_samples = np.cumsum(
            [wg.num_samples for wg in self._window_gen]
        )

        self._input_normalizer = input_normalizer
        self._target_normalizer = target_normalizer

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        input_columns: typing.Sequence[str],
        seq_len: int,
        target_columns: typing.Sequence[str] | None = None,
        stride: int = 1,
        predict: bool = False,
    ) -> TimeSeriesDataset:
        input_data = dataframe[list(input_columns)].to_numpy()

        if target_columns is None:
            target_data = None
        else:
            target_data = dataframe[list(target_columns)].to_numpy()

        return cls(
            input_data=input_data,
            seq_len=seq_len,
            target_data=target_data,
            stride=stride,
            predict=predict,
        )

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self.num_samples

    @property
    def num_samples(self) -> int:
        """
        The total number of samples in the dataset.
        :return: The number of samples.
        """
        return self._cum_num_samples[-1]

    @property
    def num_points(self) -> int:
        """
        The total number of points in the dataset.
        This is different from :meth:`num_samples` that gives the number of
        samples in the dataset. This is the number of points in the original
        dataframes.

        One element per internal dataframe.

        :return: The number of points.
        """
        return int(np.sum([len(arr) for arr in self._input_data]))

    @property
    def input_scale(self) -> np.ndarray:
        if self._input_normalizer is None:
            return np.array([])

        return self._input_normalizer.get_parameters().numpy()

    @property
    def input_scaler(self) -> RunningNormalizer | None:
        return self._input_normalizer

    @property
    def target_scale(self) -> np.ndarray:
        if self._target_normalizer is None:
            return np.array([])

        return self._target_normalizer.get_parameters().numpy()

    @property
    def target_scaler(self) -> RunningNormalizer | None:
        return self._target_normalizer

    def inverse_transform(
        self,
        input: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._input_normalizer is None:
            inv_input = input
        elif input is None:
            inv_input = None
        else:
            inv_input = self._input_normalizer.inverse_transform(input)

        if self._target_normalizer is None:
            inv_target = target
        elif target is None:
            inv_target = None
        else:
            inv_target = self._target_normalizer.inverse_transform(target)

        return inv_input, inv_target

    def __getitem__(self, idx: int) -> TimeSeriesSample:
        """
        Get a single sample from the dataset.

        :param idx: The index of the sample to get.
        :return: A tuple of the input and target torch.Tensors.
        """
        if self._dataset_type in (DataSetType.TRAIN, DataSetType.VAL_TEST):
            x, y = self._create_sample(idx)
            return {"input": x, "target": y}
        elif self._dataset_type == DataSetType.PREDICT:
            x = self._get_prediction_input(idx)
            return {"input": x}
        else:
            raise ValueError(f"Unknown dataset type {self._dataset_type}")

    def _check_index(self, idx: int) -> int:
        if idx > self.num_samples or idx < -self.num_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset with "
                f" {self.num_samples} samples."
            )

        if idx < 0:
            idx += self.num_samples

        return idx

    def _create_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a single sample from the dataset. Used internally by the __getitem__ method.

        :param idx: The index of the sample to get.

        :return: A tuple of the input and target torch.Tensors.
        """
        idx = self._check_index(idx)

        # find which df to get samples from
        df_idx = np.argmax(self._cum_num_samples > idx)

        shifted_idx = (
            idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx
        )

        x, y = self._window_gen[df_idx].get_sample(shifted_idx)

        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        return x, y

    def _get_prediction_input(self, idx: int) -> torch.Tensor:
        """
        Get a single prediction input from the dataset.

        :param idx: The index of the sample to get.
        :return: A torch.Tensor.
        """
        idx = self._check_index(idx)

        x = self._window_gen[0].get_sample(idx)
        x = torch.from_numpy(x).to(torch.float32)

        return x
