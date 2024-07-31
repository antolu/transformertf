from __future__ import annotations

import logging
import typing

import numpy as np
import torch

from .._dtype import VALID_DTYPES, convert_data
from .._sample_generator import (
    EncoderDecoderTargetSample,
    EncoderTargetSample,
)
from ._base import (
    _check_index,
)
from ._transformer import TransformerDataset

log = logging.getLogger(__name__)


RNG = np.random.default_rng()


class EncoderDataset(TransformerDataset):
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


def convert_sample(
    sample: EncoderDecoderTargetSample, dtype: VALID_DTYPES
) -> EncoderDecoderTargetSample[torch.Tensor]:
    return typing.cast(
        EncoderDecoderTargetSample[torch.Tensor],
        {k: convert_data(v, dtype=dtype)[0] for k, v in sample.items()},
    )
