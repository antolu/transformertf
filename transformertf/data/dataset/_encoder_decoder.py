from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

from .._covariates import TIME_PREFIX as TIME
from .._dtype import VALID_DTYPES, convert_data
from .._sample_generator import EncoderDecoderTargetSample
from ..transform import BaseTransform
from ._base import _check_index
from ._transformer import TransformerDataset

RND_G = np.random.default_rng()


class EncoderDecoderDataset(TransformerDataset):
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
        sample = self._apply_randomize_seq_len(sample)
        sample = self._format_time_data(sample)
        sample = apply_transforms(  # N.B. transforms also zeroed out data
            sample, self._transforms
        )

        sample_torch = convert_sample(sample, self._dtype)

        # mask zeroed out data after transforms
        sample_torch = self._apply_masks(sample_torch)

        # normalize lengths
        sample_torch["encoder_lengths"] = (
            2.0 * sample_torch["encoder_lengths"].view((1,)) / self.ctxt_seq_len - 1.0
        )
        sample_torch["decoder_lengths"] /= self.tgt_seq_len
        sample_torch["decoder_lengths"] = sample_torch["decoder_lengths"].view((1,))

        return sample_torch

    @staticmethod
    def _apply_masks(
        sample: EncoderDecoderTargetSample[torch.Tensor],
    ) -> EncoderDecoderTargetSample[torch.Tensor]:
        if "encoder_mask" in sample:
            sample["encoder_input"] *= sample["encoder_mask"]
        if "decoder_mask" in sample:
            sample["decoder_input"] *= sample["decoder_mask"]
        if "target_mask" in sample:
            sample["target"] *= sample["target_mask"]
        return sample

    def _format_time_data(
        self, sample: EncoderDecoderTargetSample[pd.DataFrame]
    ) -> EncoderDecoderTargetSample[pd.DataFrame]:
        if self._time_data and self._time_data[0] is None:
            return sample

        if TIME not in sample["encoder_input"]:
            msg = "Time column not found in encoder_input."
            raise ValueError(msg)

        seq_start = int(self.ctxt_seq_len - sample["encoder_lengths"].iloc[0].item())
        if self._time_format == "absolute":
            dt = float(sample["encoder_input"].loc[seq_start, TIME])

            # if randomize seq len, then we need to adjust the time only for the
            # nonzero values
            sample["encoder_input"].loc[seq_start:, TIME] -= dt
            sample["decoder_input"].loc[:, TIME] -= dt

            # handle zero-padded (on the right) decoder_input
            sample["decoder_input"].loc[
                sample["decoder_input"].loc[:, TIME] < 0, TIME
            ] = 0.0
        elif self._time_format == "relative":
            # first delta t is 0, to be applied wit the mask
            # sample["encoder_input"].loc[seq_start, TIME] = 0.0
            sample["encoder_mask"].loc[seq_start, TIME] = 0.0

        return sample

    def _apply_randomize_seq_len(
        self, sample: EncoderDecoderTargetSample[pd.DataFrame]
    ) -> EncoderDecoderTargetSample[pd.DataFrame]:
        if self._randomize_seq_len:
            assert self._min_ctxt_seq_len is not None
            assert self._min_tgt_seq_len is not None

            encoder_len = sample_len(self._min_ctxt_seq_len, self.ctxt_seq_len)
            to_zero = self.ctxt_seq_len - encoder_len
            sample["encoder_input"].iloc[:to_zero] = 0.0
            sample["encoder_mask"].iloc[:to_zero] = 0.0

            sample["encoder_lengths"] = pd.DataFrame({"encoder_lengths": [encoder_len]})

            decoder_len = sample_len(self._min_tgt_seq_len, self.tgt_seq_len)
            to_zero = decoder_len
            sample["decoder_input"][to_zero:] = 0.0
            sample["decoder_mask"][to_zero:] = 0.0
            sample["target"][to_zero:] = 0.0

            if "target_mask" in sample:
                sample["target_mask"][to_zero:] = 0.0

            sample["decoder_lengths"] = pd.DataFrame({"decoder_lengths": [decoder_len]})
        else:
            sample["encoder_lengths"] = pd.DataFrame({
                "encoder_lengths": [self.ctxt_seq_len],
            })
            sample["decoder_lengths"] = pd.DataFrame({
                "decoder_lengths": [self.tgt_seq_len],
            })

        # add extra dimension to target
        if "target" in sample and sample["target"].ndim == 1:
            sample["target"] = sample["target"][:, None]

        return sample


def sample_len(min_: int, max_: int) -> int:
    return int(np.round(RND_G.beta(1.0, 0.5) * (max_ - min_) + min_))


def convert_sample(
    sample: EncoderDecoderTargetSample, dtype: VALID_DTYPES
) -> EncoderDecoderTargetSample[torch.Tensor]:
    return typing.cast(
        EncoderDecoderTargetSample[torch.Tensor],
        {k: convert_data(v, dtype)[0] for k, v in sample.items()},
    )


def apply_transforms(
    sample: EncoderDecoderTargetSample[pd.DataFrame],
    transforms: typing.Mapping[str, BaseTransform] | None = None,
) -> EncoderDecoderTargetSample[pd.DataFrame]:
    """
    Apply transforms to a sample.

    Parameters
    ----------
    sample : TimeSeriesSample | EncoderTargetSample | EncoderDecoderTargetSample
    transforms : dict[str, BaseTransform] | None

    Returns
    -------
    U
    """
    if transforms is None:
        return sample

    df: pd.DataFrame
    for key in ["encoder_input", "decoder_input", "target"]:
        if key.endswith(("_mask", "_lengths")):
            continue

        df = sample[key]  # type: ignore[literal-required]
        for col in df.columns:
            if col in transforms and col != TIME:
                transform = transforms[col]
                if transform.transform_type == transform.TransformType.XY:
                    msg = "Cannot do two-variable transforms on a Dataset level (yet)."
                    raise NotImplementedError(msg)

                sample[key][col] = transform.transform(df[col].to_numpy()).numpy()  # type: ignore[literal-required]

    encoder_len = len(sample["encoder_input"])

    if (
        TIME in sample["encoder_input"]
        and TIME in sample["decoder_input"]
        and TIME in transforms
    ):
        time = np.concatenate([
            sample["encoder_input"][TIME].to_numpy(),
            sample["decoder_input"][TIME].to_numpy(),
        ])
        time = transforms[TIME].transform(time).numpy()
        sample["encoder_input"][TIME] = time[:encoder_len]
        sample["decoder_input"][TIME] = time[encoder_len:]

    return sample
