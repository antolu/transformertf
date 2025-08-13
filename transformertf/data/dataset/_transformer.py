from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd

from .._dtype import VALID_DTYPES
from .._window_strategy import TransformerWindowStrategy
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
        stride: int = 1,
        noise_std: float = 0.0,
        predict: bool = False,
        apply_transforms: bool = True,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        add_target_to_past: bool = True,
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
        masked_encoder_features: list[str] | None = None,
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

        if predict and randomize_seq_len:
            # TODO: allow random seq len for validation purposes
            log.warning("randomize_seq_len is ignored when predicting.")

            randomize_seq_len = False

        # Create window strategy to handle sample generator creation
        window_strategy = TransformerWindowStrategy(
            ctx_seq_len=ctx_seq_len,
            tgt_seq_len=tgt_seq_len,
            stride=stride,
            min_ctx_seq_len=min_ctxt_seq_len,
            min_tgt_seq_len=min_tgt_seq_len,
            randomize_seq_len=randomize_seq_len,
            predict=predict,
        )

        self._ctxt_seq_len = ctx_seq_len
        self._min_ctxt_seq_len = min_ctxt_seq_len
        self._tgt_seq_len = tgt_seq_len
        self._min_tgt_seq_len = min_tgt_seq_len
        self._predict = predict
        self._stride = stride
        self._randomize_seq_len = randomize_seq_len
        self._add_target_to_past = add_target_to_past
        self._dtype = dtype
        self._time_format = time_format
        self._apply_transforms = apply_transforms
        if noise_std < 0.0:
            msg = "noise_std must be non-negative"
            raise ValueError(msg)
        self._noise_std = noise_std

        self._transforms = transforms or {}
        self._masked_encoder_features = masked_encoder_features or []

        self._sample_gen = window_strategy.create_sample_generators(
            input_data=self._input_data,
            target_data=self._target_data,
            known_past_data=self._known_past_data,
            time_data=self._time_data,
            add_target_to_past=add_target_to_past,
        )

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
