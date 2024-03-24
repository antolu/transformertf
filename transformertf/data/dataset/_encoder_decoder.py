from __future__ import annotations


import numpy as np


from ._base import (
    _check_index,
)
from ._encoder import EncoderDataset
from .._sample_generator import EncoderDecoderTargetSample


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
            sample["encoder_input"][:random_len] = 0.0

            random_len = np.random.randint(
                self._min_tgt_seq_len, self._tgt_seq_len
            )
            sample["decoder_input"][random_len:] = 0.0

            if "target" in sample:
                sample["target"][random_len:] = 0.0

        return sample
