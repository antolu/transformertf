from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import torch

from .._sample_generator import EncoderTargetSample, TransformerSampleGenerator
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
        self._min_ctxt_seq_len = min_ctxt_seq_len
        self._tgt_seq_len = tgt_seq_len
        self._min_tgt_seq_len = min_tgt_seq_len
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
        self._dtype = dtype

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
    ) -> EncoderDataset:
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

    def __getitem__(self, idx: int) -> EncoderTargetSample:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx

        Returns
        -------
        EncoderTargetSample
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
            assert self._min_tgt_seq_len is not None
            random_len = np.random.randint(
                self._min_ctxt_seq_len, self._ctxt_seq_len
            )
            sample["encoder_input"][: self._ctxt_seq_len - random_len] = 0.0

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
