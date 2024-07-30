from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import torch

from .._dtype import VALID_DTYPES, convert_data
from .._sample_generator import (
    EncoderDecoderTargetSample,
    EncoderTargetSample,
    TransformerSampleGenerator,
)
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


class EncoderDataset(AbstractTimeSeriesDataset):
    def __init__(
        self,
        input_data: pd.DataFrame | list[pd.DataFrame],
        target_data: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame],
        ctx_seq_len: int,
        tgt_seq_len: int,
        *,
        known_past_data: pd.DataFrame | list[pd.DataFrame] | None = None,
        stride: int = 1,
        predict: bool = False,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        input_transforms: dict[str, BaseTransform] | None = None,
        target_transform: BaseTransform | None = None,
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
                msg = (
                    "min_tgt_seq_len must be specified when "
                    "randomize_seq_len is True"
                )
                raise ValueError(msg)
            if min_ctxt_seq_len is None:
                msg = (
                    "min_ctx_seq_len must be specified when "
                    "randomize_seq_len is True"
                )
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

        self._dataset_type = DataSetType.VAL_TEST if predict else DataSetType.TRAIN

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
        self._dtype = dtype

        self._input_transforms = input_transforms or {}
        self._target_transform = target_transform

        self._sample_gen = [
            TransformerSampleGenerator(
                input_data=input_,
                target_data=target_,
                known_past_data=known_,
                src_seq_len=ctx_seq_len,
                tgt_seq_len=tgt_seq_len,
                zero_pad=predict,
                stride=stride,
            )
            for input_, target_, known_ in zip(
                self._input_data,
                self._target_data,
                self._known_past_data,
                strict=False,
            )
        ]

        self._cum_num_samples = np.cumsum([len(gen) for gen in self._sample_gen])
        self._dtype = dtype

    def __getitem__(self, idx: int) -> EncoderTargetSample[torch.Tensor]:
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
        shifted_idx = idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx

        sample: EncoderDecoderTargetSample = self._sample_gen[df_idx][shifted_idx]

        if self._randomize_seq_len:
            assert self._min_ctxt_seq_len is not None
            assert self._min_tgt_seq_len is not None
            random_len = RNG.integers(self._min_ctxt_seq_len, self._ctxt_seq_len)
            sample["encoder_input"].iloc[: self._ctxt_seq_len - random_len] = 0.0

            random_len = RNG.integers(self._min_tgt_seq_len, self._tgt_seq_len)
            sample["decoder_input"].iloc[random_len:] = 0.0

            if "target" in sample:
                sample["target"].iloc[random_len:] = 0.0

        sample_torch = convert_sample(sample, self._dtype)

        # concatenate input and target data
        target_old = sample_torch["encoder_input"][..., -1, None]

        target = torch.concat((target_old, sample_torch["target"]), dim=0)
        encoder_input = torch.concat(
            (sample_torch["encoder_input"], sample_torch["decoder_input"]),
            dim=0,
        )

        return typing.cast(
            EncoderTargetSample[torch.Tensor],
            {
                "encoder_input": encoder_input,
                "encoder_mask": torch.ones_like(encoder_input),
                "target": target,
            },
        )

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


def convert_sample(
    sample: EncoderDecoderTargetSample, dtype: VALID_DTYPES
) -> EncoderDecoderTargetSample[torch.Tensor]:
    return typing.cast(
        EncoderDecoderTargetSample[torch.Tensor],
        {k: convert_data(v, dtype=dtype)[0] for k, v in sample.items()},
    )
