from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import torch

from .._dtype import VALID_DTYPES, convert_data
from .._sample_generator import TimeSeriesSample, TimeSeriesSampleGenerator
from ..transform import BaseTransform
from ._base import (
    AbstractTimeSeriesDataset,
    DataSetType,
    _check_index,
    _check_label_data_length,
    _to_list,
)

log = logging.getLogger(__name__)


RNG = np.random.default_rng()


class TimeSeriesDataset(AbstractTimeSeriesDataset):
    _input_data: list[pd.DataFrame]
    _target_data: list[pd.DataFrame] | list[None]

    def __init__(
        self,
        input_data: pd.DataFrame | list[pd.DataFrame],
        seq_len: int | None = None,
        target_data: pd.DataFrame
        | pd.Series
        | list[pd.Series]
        | list[pd.DataFrame]
        | None = None,
        *,
        stride: int = 1,
        predict: bool = False,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
    ):
        """
        Time series dataset for training, validation, and prediction. The dataset
        takes one or more dataframes or numpy arrays as input. The input data
        should have dimension [num_points, num_features]. For instance, with one
        univariate time series, the input should have dimension [num_points, 1].
        The target data is optional and can be used for training and validation. If
        the target data is provided, it should have the same number of points as the
        input data. The target data can be a numpy array or a pandas DataFrame.

        Multiple dataframes can be provided, and the class will calculate
        samples from each dataframe on-demand in the __getitem__ method. There must
        within each pair of input/target dataframes be the same number of points, and
        there must be a one-to-one correspondence between the dataframes.

        This is done to save memory on larger datasets.
        The dataframes are never merged/concatenated into a single dataset
        as the gap between the timestamps in the dataframes represent
        a discontinuity in the data with different time series.

        The samples are created by creating "sliding windows" from the input
        and the target data.
        The dimensions of the samples created using the
        :method:`__getitem__` method are (seq_len, num_features). If there is labeled data,
        the target data is also included in the samples and will have dimensions
        (seq_len, 1).
        This is concatenated into a single :class:`torch.Tensor` of shape
        (batch, seq_len, 1 or 2) during training. The batch dimension is added by the
        :class:`torch.utils.data.DataLoader` class.

        **N.B.** The input and target transforms are not applied to the dataframes,
        and are only encapsulated in object for optional postprocessing use by the
        user.

        Parameters
        ----------
        input_data : np.ndarray | pd.DataFrame | list[np.ndarray | pd.DataFrame]
            The input data for the dataset. If multiple data sources are used,
            provide a list of data sources. Data should already be transformed,
            since the dataset will only cut the data into samples.
        target_data : np.ndarray | pd.DataFrame | list[np.ndarray | pd.DataFrame], optional
            The target data for the dataset. If multiple data sources are used,
            provide a list of data sources. Data should already be transformed,
            since the dataset will only cut the data into samples.
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
        transforms : dict[str, BaseTransform], optional
            The transforms that were used to transform the data.
            This can be used to inverse transform the data.
        dtype : torch.dtype
            The data type of the torch.Tensors returned by the dataset in the
            __getitem__ method. Defaults to torch.float32.
        """
        super().__init__()

        if randomize_seq_len:
            if seq_len is None:
                msg = "seq_len must be specified when randomize_seq_len is True."
                raise ValueError(msg)
            if min_seq_len is None:
                msg = "min_seq_len must be specified when randomize_seq_len is True."
                raise ValueError(msg)
            if min_seq_len > seq_len:
                msg = "min_seq_len must be less than or equal to seq_len."
                raise ValueError(msg)

        self._input_data = _to_list(input_data)

        if target_data is not None or (
            isinstance(target_data, list) and len(target_data) > 0
        ):
            self._target_data = typing.cast(
                list[pd.DataFrame],
                [
                    df.to_frame() if isinstance(df, pd.Series) else df
                    for df in _to_list(target_data)
                ],
            )

            _check_label_data_length(self._input_data, self._target_data)

            # if there is labeled data, it's either for training or validation
            if predict:
                self._dataset_type = DataSetType.VAL_TEST
            else:
                self._dataset_type = DataSetType.TRAIN

        else:  # no label predict
            if not predict:
                msg = "Cannot use predict=False with no label data."
                raise ValueError(msg)

            self._target_data = [None]
            self._dataset_type = DataSetType.PREDICT

            if len(self._input_data) > 1:
                msg = (
                    f"Predicting requires exactly one input data source. "
                    f"Got {len(self._input_data)}"
                )
                raise ValueError(msg)

        if seq_len is None:
            if len(self._input_data) > 1:
                msg = (
                    "seq_len must be specified when using more than one input "
                    f"data source. Got {len(self._input_data)} input data sources."
                )
                raise ValueError(msg)
            seq_len = len(self._input_data[0])
            log.info(f"Using full dataset as sequence length: {seq_len}")

        if predict:
            if stride != 1:
                log.warning("Stride is ignored when predicting.")
            stride = seq_len

        self._seq_len = seq_len
        self._min_seq_len = min_seq_len
        self._randomize_seq_len = randomize_seq_len
        self._stride = stride
        self._predict = predict
        self._dtype = dtype

        self._transforms = transforms or {}

        self._sample_gen: list[TimeSeriesSampleGenerator[pd.DataFrame]] = [
            TimeSeriesSampleGenerator(
                input_data=input_,
                window_size=seq_len,
                label_data=target,
                stride=stride,
                zero_pad=predict,
            )
            for input_, target in zip(self._input_data, self._target_data, strict=False)
        ]

        # needed to determine which dataframe to get samples from
        self._cum_num_samples = np.cumsum([len(gen) for gen in self._sample_gen])

    def __getitem__(self, idx: int) -> TimeSeriesSample[torch.Tensor]:
        """
        Get a single sample from the dataset.

        :param idx: The index of the sample to get.
        :return: A tuple of the input and target torch.Tensors.
        """
        if self._dataset_type in {DataSetType.TRAIN, DataSetType.VAL_TEST}:
            return self._create_sample(idx)
        if self._dataset_type == DataSetType.PREDICT:
            return self._get_prediction_input(idx)
        msg = f"Unknown dataset type {self._dataset_type}"
        raise ValueError(msg)

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return int(self._cum_num_samples[-1])

    def __iter__(self) -> typing.Iterator[TimeSeriesSample]:
        for i in range(len(self)):
            yield self[i]

    def _create_sample(self, idx: int) -> TimeSeriesSample[torch.Tensor]:
        """
        Create a single sample from the dataset. Used internally by the __getitem__ method.

        :param idx: The index of the sample to get.

        :return: A tuple of the input and target torch.Tensors.
        """
        idx = _check_index(idx, len(self))

        # find which df to get samples from
        df_idx = np.argmax(self._cum_num_samples > idx)

        shifted_idx = idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx

        sample: TimeSeriesSample = self._sample_gen[df_idx][shifted_idx]

        if self._randomize_seq_len:
            assert self._min_seq_len is not None
            random_len = RNG.integers(self._min_seq_len, self._seq_len)
            sample["input"].iloc[random_len:] = 0.0

            if "target" in sample:
                sample["target"].iloc[random_len:] = 0.0

        sample_torch = convert_sample(sample, self._dtype)

        # add dimension for num features
        if sample_torch["input"].ndim == 1:
            sample_torch["input"] = sample_torch["input"][..., None]
        if "target" in sample_torch and sample_torch["target"].ndim == 1:
            sample_torch["target"] = sample_torch["target"][..., None]

        return sample_torch

    def _get_prediction_input(self, idx: int) -> TimeSeriesSample[torch.Tensor]:
        """
        Get a single prediction input from the dataset.

        :param idx: The index of the sample to get.
        :return: A torch.Tensor.
        """
        idx = _check_index(idx, len(self))

        x = self._sample_gen[0][idx]

        if self._randomize_seq_len:
            assert self._min_seq_len is not None
            random_len = RNG.integers(self._min_seq_len, self._seq_len)
            x["input"].iloc[random_len:] = 0.0

        return convert_sample(x, self._dtype)


def convert_sample(
    sample: TimeSeriesSample, dtype: VALID_DTYPES
) -> TimeSeriesSample[torch.Tensor]:
    """
    Convert the data in a sample to a torch.Tensor with the given dtype.
    """
    sample_torch = {}
    for key, value in sample.items():
        sample_torch[key] = convert_data(value, dtype)[0]

    return typing.cast(TimeSeriesSample[torch.Tensor], sample_torch)
