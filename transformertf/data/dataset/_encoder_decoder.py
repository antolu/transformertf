from __future__ import annotations

import numpy as np
import torch

from .._sample_generator import EncoderDecoderTargetSample
from ._base import _check_index, get_dtype
from ._encoder import EncoderDataset

RND_G = np.random.default_rng()


class EncoderDecoderDataset(EncoderDataset):
    def __getitem__(self, idx: int) -> EncoderDecoderTargetSample:  # type: ignore[override]
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
        df_idx = np.argmax(self._cum_num_samples > idx)

        shifted_idx = idx - self._cum_num_samples[df_idx - 1] if df_idx > 0 else idx

        sample = self._sample_gen[df_idx][shifted_idx]

        if self._randomize_seq_len:
            assert self._min_ctxt_seq_len is not None
            assert self._min_tgt_seq_len is not None
            encoder_len = sample_len(self._min_ctxt_seq_len, self.ctxt_seq_len)
            sample["encoder_input"][: self.ctxt_seq_len - encoder_len] = 0.0
            sample["encoder_mask"][: self.ctxt_seq_len - encoder_len] = 0.0

            encoder_len_ = 2.0 * encoder_len / self.ctxt_seq_len - 1.0
            sample["encoder_lengths"] = torch.tensor(
                [encoder_len_], dtype=get_dtype(self._dtype)
            )

            decoder_len = sample_len(self._min_tgt_seq_len, self.tgt_seq_len)
            sample["decoder_input"][: self.tgt_seq_len - decoder_len] = 0.0
            sample["decoder_mask"][: self.tgt_seq_len - decoder_len] = 0.0
            sample["target"][: self.tgt_seq_len - decoder_len] = 0.0

            decoder_len_ = decoder_len / self.tgt_seq_len
            sample["decoder_lengths"] = torch.tensor(
                [decoder_len_], dtype=get_dtype(self._dtype)
            )
        else:
            sample["encoder_lengths"] = torch.tensor(
                [1.0], dtype=get_dtype(self._dtype)
            )
            sample["decoder_lengths"] = torch.tensor(
                [1.0], dtype=get_dtype(self._dtype)
            )

        return sample


def sample_len(min_: int, max_: int) -> int:
    return int(np.round(RND_G.beta(1.0, 0.5) * (max_ - min_) + min_))
