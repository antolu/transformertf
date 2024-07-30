from __future__ import annotations

import sys
import typing

import numpy as np
import torch

from .._dtype import VALID_DTYPES, convert_data, get_dtype
from .._sample_generator import EncoderDecoderTargetSample
from ._base import _check_index
from ._encoder import EncoderDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

RND_G = np.random.default_rng()


class EncoderDecoderDataset(EncoderDataset):
    @override
    def __getitem__(self, idx: int) -> EncoderDecoderTargetSample[torch.Tensor]:  # type: ignore[override]
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx

        Returns
        -------
        EncoderDecoderTargetSample
        """
        idx = _check_index(idx, len(self))

        # find which df to get samples from
        df_idx = int(np.argmax(self._cum_num_samples > idx))
        shifted_idx = int(
            idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx
        )

        sample = self._sample_gen[df_idx][shifted_idx]

        sample_torch = convert_sample(sample, self._dtype)

        if self._randomize_seq_len:
            assert self._min_ctxt_seq_len is not None
            assert self._min_tgt_seq_len is not None
            encoder_len = sample_len(self._min_ctxt_seq_len, self.ctxt_seq_len)
            sample_torch["encoder_input"][: self.ctxt_seq_len - encoder_len] = 0.0
            sample_torch["encoder_mask"][: self.ctxt_seq_len - encoder_len] = 0.0

            encoder_len_ = 2.0 * encoder_len / self.ctxt_seq_len - 1.0
            sample_torch["encoder_lengths"] = torch.tensor(
                [encoder_len_], dtype=get_dtype(self._dtype)
            )

            decoder_len = sample_len(self._min_tgt_seq_len, self.tgt_seq_len)
            sample_torch["decoder_input"][: self.tgt_seq_len - decoder_len] = 0.0
            sample_torch["decoder_mask"][: self.tgt_seq_len - decoder_len] = 0.0
            sample_torch["target"][: self.tgt_seq_len - decoder_len] = 0.0

            decoder_len_ = decoder_len / self.tgt_seq_len
            sample_torch["decoder_lengths"] = torch.tensor(
                [decoder_len_], dtype=get_dtype(self._dtype)
            )
        else:
            sample_torch["encoder_lengths"] = torch.tensor(
                [1.0], dtype=get_dtype(self._dtype)
            )
            sample_torch["decoder_lengths"] = torch.tensor(
                [1.0], dtype=get_dtype(self._dtype)
            )
        if "target" in sample_torch and sample_torch["target"].ndim == 1:
            sample_torch["target"] = sample_torch["target"][:, None]

        return sample_torch


def sample_len(min_: int, max_: int) -> int:
    return int(np.round(RND_G.beta(1.0, 0.5) * (max_ - min_) + min_))


def convert_sample(
    sample: EncoderDecoderTargetSample, dtype: VALID_DTYPES
) -> EncoderDecoderTargetSample[torch.Tensor]:
    return typing.cast(
        EncoderDecoderTargetSample[torch.Tensor],
        {k: convert_data(v, dtype)[0] for k, v in sample.items()},
    )
