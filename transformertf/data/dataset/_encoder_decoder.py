from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

from .._covariates import TIME_PREFIX as TIME
from .._dtype import VALID_DTYPES, convert_data
from .._sample_generator import (
    DecoderSample,
    EncoderDecoderSample,
    EncoderDecoderTargetSample,
    EncoderSample,
)
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
        sample = self._format_time_data(
            sample,
            time_format=self._time_format,
            ctxt_seq_len=self.ctxt_seq_len,
        )
        sample = apply_transforms(  # N.B. transforms also zeroed out data
            sample, transforms=self._transforms
        )

        sample_torch = convert_sample(sample, self._dtype)

        # add noise
        if self._noise_std > 0.0:
            skip = 1 if TIME in sample["encoder_input"] else 0
            sample_torch["encoder_input"][..., skip:] += torch.normal(
                mean=0.0,
                std=torch.ones_like(sample_torch["encoder_input"][..., skip:])
                * self._noise_std,
            )

            skip = 1 if TIME in sample["decoder_input"] else 0
            sample_torch["decoder_input"][..., skip:-1] += torch.normal(
                mean=0.0,
                std=torch.ones_like(sample_torch["decoder_input"][..., skip:-1])
                * self._noise_std,
            )

        # mask zeroed out data after transforms
        sample_torch = self._apply_masks(sample_torch)

        # add extra dimension to target
        if "target" in sample_torch and sample_torch["target"].ndim == 1:
            sample_torch["target"] = sample_torch["target"][:, None]

        if TIME in sample["encoder_input"] and self._time_format == "relative":
            sample_torch["encoder_input"][
                -int(sample_torch.get("encoder_lengths", self.ctxt_seq_len)), 0
            ] = 0.0

        # normalize lengths
        # sample_torch["encoder_lengths"] = (
        #     2.0
        #     * sample_torch.get(
        #         "encoder_lengths", torch.tensor(self.ctxt_seq_len)
        #     ).view((1,))
        #     / self.ctxt_seq_len
        #     - 1.0
        # )
        # sample_torch["decoder_lengths"] /= self.tgt_seq_len
        sample_torch["encoder_lengths"] = sample_torch["encoder_lengths"].view((1,))
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

    @staticmethod
    @typing.overload
    def _format_time_data(
        sample: EncoderSample[pd.DataFrame],
        time_format: typing.Literal["absolute", "relative"] = "relative",
        ctxt_seq_len: int | None = None,
        *,
        encoder: bool = True,
        decoder: bool = False,
    ) -> EncoderSample[pd.DataFrame]: ...

    @staticmethod
    @typing.overload
    def _format_time_data(
        sample: EncoderDecoderSample[pd.DataFrame],
        time_format: typing.Literal["absolute", "relative"] = "relative",
        ctxt_seq_len: int | None = None,
        *,
        encoder: bool = True,
        decoder: bool = True,
    ) -> EncoderDecoderSample[pd.DataFrame]: ...

    @staticmethod
    @typing.overload  # type ignore[misc]
    def _format_time_data(  # type: ignore[misc]
        sample: EncoderDecoderTargetSample[pd.DataFrame],
        time_format: typing.Literal["absolute", "relative"] = "relative",
        ctxt_seq_len: int | None = None,
        *,
        encoder: bool = True,
        decoder: bool = True,
    ) -> EncoderDecoderTargetSample[pd.DataFrame]: ...

    @staticmethod
    @typing.overload
    def _format_time_data(
        sample: DecoderSample[pd.DataFrame],
        time_format: typing.Literal["absolute", "relative"] = "relative",
        ctxt_seq_len: int | None = None,
        *,
        encoder: bool = False,
        decoder: bool = True,
    ) -> DecoderSample[pd.DataFrame]: ...

    @staticmethod
    def _format_time_data(  # type: ignore
        sample: EncoderSample[pd.DataFrame]
        | EncoderDecoderSample[pd.DataFrame]
        | EncoderDecoderTargetSample[pd.DataFrame]
        | DecoderSample[pd.DataFrame],
        time_format: typing.Literal["absolute", "relative"] = "relative",
        ctxt_seq_len: int | None = None,
        *,
        encoder: bool = True,
        decoder: bool = True,
    ) -> (
        EncoderSample[pd.DataFrame]
        | EncoderDecoderSample[pd.DataFrame]
        | EncoderDecoderTargetSample[pd.DataFrame]
        | DecoderSample[pd.DataFrame]
    ):
        if (encoder and TIME not in sample["encoder_input"]) or (  # type: ignore[typeddict-item]
            decoder and TIME not in sample["decoder_input"]  # type: ignore[typeddict-item]
        ):
            return sample

        if encoder:
            ctxt_seq_len = ctxt_seq_len or len(sample["encoder_input"])  # type: ignore[typeddict-item]

            ctxt_start = int(ctxt_seq_len - sample["encoder_lengths"].iloc[0, 0].item())  # type: ignore[typeddict-item, union-attr, operator, arg-type]
        else:
            ctxt_start = 0

        if time_format == "absolute":
            if encoder:
                dt = float(sample["encoder_input"].loc[ctxt_start, TIME])  # type: ignore[arg-type, typeddict-item]
            else:
                dt = float(sample["decoder_input"].loc[0, TIME])  # type: ignore[arg-type, typeddict-item]

            # if randomize seq len, then we need to adjust the time only for the
            # nonzero values
            if encoder:
                sample["encoder_input"].loc[ctxt_start:, TIME] -= dt  # type: ignore[typeddict-item]
            if decoder:
                sample["decoder_input"].loc[:, TIME] -= dt  # type: ignore[typeddict-item]

                # handle zero-padded (on the right) decoder_input
                sample["decoder_input"].loc[  # type: ignore[typeddict-item]
                    sample["decoder_input"].loc[:, TIME] < 0, TIME  # type: ignore[typeddict-item]
                ] = 0.0

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

        return sample

    @staticmethod
    def make_encoder_input(
        df: pd.DataFrame,
        seq_len: int | None = None,
        time_data: pd.Series | pd.DataFrame | None = None,
        time_format: typing.Literal["absolute", "relative"] = "relative",
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
    ) -> EncoderSample[torch.Tensor]:
        """
        Make the encoder input from the DataFrame. The whole DataFrame is used.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to make encoder input from.
        time_format

        Returns
        -------
        pd.DataFrame
        """
        if seq_len is not None:
            if seq_len > len(df):
                msg = (
                    "seq_len must be less than or equal to the length of the DataFrame."
                )
                raise ValueError(msg)

            df = df.iloc[-seq_len:]

        # if TIME not in df:
        #     if time_data is None:
        #         msg = "Time column not found in DataFrame and time_data is None."
        #         raise ValueError(msg)
        #     df[TIME] = time_data.to_numpy()

        df = df.reset_index(drop=True)

        sample: EncoderSample[pd.DataFrame] = {
            "encoder_input": df,
            "encoder_mask": pd.DataFrame(np.ones((len(df), 1))),
            "encoder_lengths": pd.DataFrame({"encoder_lengths": [len(df)]}),
        }

        # shift time to start with 0
        sample = EncoderDecoderDataset._format_time_data(
            sample,
            time_format=time_format,
            ctxt_seq_len=len(df),
            decoder=False,
        )
        sample = apply_transforms(sample, transforms=transforms)  # type: ignore[arg-type]

        return convert_sample(sample, dtype)  # type: ignore[return-value]

    @staticmethod
    def make_decoder_input(
        df: pd.DataFrame,
        seq_len: int | None = None,
        time_data: pd.Series | pd.DataFrame | None = None,
        time_format: typing.Literal["absolute", "relative"] = "relative",
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
    ) -> DecoderSample[torch.Tensor]:
        """
        Make the decoder input from the DataFrame. The whole DataFrame is used.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to make decoder input from.
        time_format

        Returns
        -------
        pd.DataFrame
        """
        if seq_len is not None:
            if seq_len > len(df):
                msg = (
                    "seq_len must be less than or equal to the length of the DataFrame."
                )
                raise ValueError(msg)

            df = df.iloc[-seq_len:]

        # if TIME not in df:
        #     if time_data is None:
        #         msg = "Time column not found in DataFrame and time_data is None."
        #         raise ValueError(msg)
        #     df[TIME] = time_data.to_numpy()

        df = df.reset_index(drop=True)

        sample: DecoderSample[pd.DataFrame] = {
            "decoder_input": df,
            "decoder_mask": pd.DataFrame(np.ones((len(df), 1))),
            "decoder_lengths": pd.DataFrame({"decoder_lengths": [len(df)]}),
        }

        # shift time to start with 0
        sample = EncoderDecoderDataset._format_time_data(
            sample,
            time_format=time_format,
            ctxt_seq_len=len(df),
            encoder=False,
        )
        sample = apply_transforms(sample, transforms=transforms, encoder=False)  # type: ignore[arg-type]

        return convert_sample(sample, dtype)


def sample_len(min_: int, max_: int) -> int:
    return int(RND_G.uniform(min_, max_))


def convert_sample(
    sample: EncoderDecoderTargetSample, dtype: VALID_DTYPES
) -> EncoderDecoderTargetSample[torch.Tensor]:
    return typing.cast(
        EncoderDecoderTargetSample[torch.Tensor],
        {k: convert_data(v, dtype)[0] for k, v in sample.items()},
    )


def apply_transforms(
    sample: EncoderDecoderTargetSample[pd.DataFrame],
    *,
    encoder: bool = True,
    decoder: bool = True,
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
    keys = []
    if encoder:
        keys.append("encoder_input")
    if decoder:
        keys.extend(["decoder_input"])
        if "target" in sample["decoder_input"].columns:
            keys.append("target")
    for key in keys:
        df = sample[key]  # type: ignore[literal-required]
        for col in df.columns:
            if col in transforms and col != TIME:
                transform = transforms[col]
                if transform.transform_type == transform.TransformType.XY:
                    msg = "Cannot do two-variable transforms on a Dataset level (yet)."
                    raise NotImplementedError(msg)

                sample[key][col] = transform.transform(df[col].to_numpy()).numpy()  # type: ignore[literal-required]

    if encoder and TIME in sample["encoder_input"] and TIME in transforms and decoder:
        encoder_len = len(sample["encoder_input"])

        if decoder and TIME in sample["decoder_input"]:
            # transform time for both encoder and decoder and ensure
            # that the time is continuous
            time = np.concatenate([
                sample["encoder_input"][TIME].to_numpy(),
                sample["decoder_input"][TIME].to_numpy(),
            ])
            time = transforms[TIME].transform(time).numpy()
            sample["encoder_input"][TIME] = time[:encoder_len]
            sample["decoder_input"][TIME] = time[encoder_len:]
        else:
            # transform time for encoder only
            sample["encoder_input"][TIME] = (
                transforms[TIME]
                .transform(sample["encoder_input"][TIME].to_numpy())
                .numpy()
            )
    elif decoder and TIME in sample["decoder_input"]:
        # transform time for decoder only
        if decoder and TIME in sample["decoder_input"] and TIME in transforms:
            sample["decoder_input"][TIME] = (
                transforms[TIME]
                .transform(sample["decoder_input"][TIME].to_numpy())
                .numpy()
            )

    return sample
