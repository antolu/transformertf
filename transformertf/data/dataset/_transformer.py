from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd

from .._dtype import VALID_DTYPES
from .._sample_generator import (
    TransformerSampleGenerator,
)
from ..transform import BaseTransform
from ._base import (
    AbstractTimeSeriesDataset,
    DataSetType,
    _check_label_data_length,
    _to_list,
)

log = logging.getLogger(__name__)


RNG = np.random.default_rng()


class TransformerDataset(AbstractTimeSeriesDataset):
    def __init__(
        self,
        input_data: pd.DataFrame | list[pd.DataFrame],
        target_data: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame],
        ctx_seq_len: int,
        tgt_seq_len: int,
        *,
        known_past_data: pd.DataFrame | list[pd.DataFrame] | None = None,
        time_data: pd.Series
        | pd.DataFrame
        | list[pd.Series | pd.DataFrame]
        | None = None,
        time_format: typing.Literal["absolute", "relative"] = "relative",
        sample_stride: int = 1,
        timestep_stride: int = 1,
        predict: bool = False,
        apply_transforms: bool = True,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
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
                msg = "min_tgt_seq_len must be specified when randomize_seq_len is True"
                raise ValueError(msg)
            if min_ctxt_seq_len is None:
                msg = "min_ctx_seq_len must be specified when randomize_seq_len is True"
                raise ValueError(msg)

        self._input_data = _to_list(input_data)
        self._target_data = typing.cast(
            list[pd.DataFrame],
            [
                o.to_frame() if isinstance(o, pd.Series) else o
                for o in _to_list(target_data)
            ],
        )
        _check_label_data_length(self._input_data, self._target_data)
        self._known_past_data = typing.cast(
            list[pd.DataFrame] | list[None],
            _to_list(known_past_data)
            if known_past_data is not None
            else [None] * len(self._input_data),
        )
        self._time_data = typing.cast(
            list[pd.DataFrame] | list[None],
            [
                o.to_frame() if isinstance(o, pd.Series) else o
                for o in _to_list(time_data)
            ]
            if time_data is not None
            else [None] * len(self._input_data),
        )

        self._dataset_type = DataSetType.VAL_TEST if predict else DataSetType.TRAIN

        if predict:
            if sample_stride != 1:
                log.warning("Stride is ignored when predicting.")
            sample_stride = tgt_seq_len
            if randomize_seq_len:
                # TODO: allow random seq len for validation purposes
                log.warning("randomize_seq_len is ignored when predicting.")

                randomize_seq_len = False

        if timestep_stride > 1:
            self._input_data = [
                df.iloc[start::timestep_stride]
                for df in self._input_data
                for start in range(timestep_stride)
            ]
            self._target_data = [
                df.iloc[start::timestep_stride]
                for df in self._target_data
                for start in range(timestep_stride)
            ]
            self._known_past_data = [  # type: ignore[assignment]
                df.iloc[start::timestep_stride] if df is not None else None
                for df in self._known_past_data
                for start in range(timestep_stride)
            ]
            self._time_data = [  # type: ignore[assignment]
                df.iloc[start::timestep_stride] if df is not None else None
                for df in self._time_data
                for start in range(timestep_stride)
            ]

        self._ctxt_seq_len = ctx_seq_len
        self._min_ctxt_seq_len = min_ctxt_seq_len
        self._tgt_seq_len = tgt_seq_len
        self._min_tgt_seq_len = min_tgt_seq_len
        self._predict = predict
        self._stride = sample_stride
        self._randomize_seq_len = randomize_seq_len
        self._dtype = dtype
        self._time_format = time_format
        self._apply_transforms = apply_transforms

        self._transforms = transforms or {}

        self._sample_gen = [
            TransformerSampleGenerator(
                input_data=pd.concat([time_, input_], axis=1)
                if time_ is not None
                else input_,
                target_data=target_,
                known_past_data=known_,
                src_seq_len=ctx_seq_len,
                tgt_seq_len=tgt_seq_len,
                zero_pad=predict,
                stride=sample_stride,
            )
            for input_, target_, known_, time_ in zip(
                self._input_data,
                self._target_data,
                self._known_past_data,
                self._time_data,
                strict=False,
            )
        ]

        self._cum_num_samples = np.cumsum([len(gen) for gen in self._sample_gen])
        self._dtype = dtype

    @property
    def ctxt_seq_len(self) -> int:
        return self._ctxt_seq_len

    @property
    def tgt_seq_len(self) -> int:
        return self._tgt_seq_len

    @property
    def stride(self) -> int:
        return self._stride

    def __len__(self) -> int:
        return int(self._cum_num_samples[-1])
