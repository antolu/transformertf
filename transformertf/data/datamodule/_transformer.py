from __future__ import annotations

import sys
import typing

import numba
import numba.typed
import numpy as np
import pandas as pd
import torch

from .._downsample import DOWNSAMPLE_METHODS
from .._dtype import VALID_DTYPES
from .._sample_generator import EncoderDecoderTargetSample
from .._window_generator import WindowGenerator
from ..dataset import EncoderDataset, EncoderDecoderDataset
from ..transform import (
    BaseTransform,
    DeltaTransform,
    MaxScaler,
    StandardScaler,
    TransformCollection,
)
from ._base import TIME_PREFIX as TIME
from ._base import DataModuleBase, _to_list

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if typing.TYPE_CHECKING:
    from ..transform import BaseTransform


class TransformerDataModule(DataModuleBase):
    def __init__(
        self,
        *,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        known_past_covariates: str | typing.Sequence[str] | None = None,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        normalize: bool = True,
        ctxt_seq_len: int = 500,
        tgt_seq_len: int = 300,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        noise_std: float = 0.0,
        target_depends_on: str | None = None,
        time_column: str | None = None,
        time_format: typing.Literal[
            "relative", "absolute", "relative_legacy"
        ] = "absolute",
        add_target_to_past: bool = True,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: VALID_DTYPES = "float32",
        shuffle: bool = True,
        distributed: bool | typing.Literal["auto"] = "auto",
    ):
        """
        For documentation of arguments see :class:`DataModuleBase`.

        time_column: str | None
            The column in the data that contains the timestamps. If None, the data is
            assumed to be evenly spaced, and no temporal features are added to the
            samples. If a column name is provided, a temporal feature will be added to
            the datasets. If the data type is datetime, the time is converted to
            milliseconds since the epoch. Time is always relative to the first timestamp
            and its magnitude does not matter, as the data is normalized prior to
            training.
        time_format : typing.Literal["relative", "absolute"]
            The format of the timestamps. If "relative", the timestamps are relative to
            the first timestamp in the data, i.e. $\\Delta$. If "absolute", the
            timestamps are in absolute time, i.e. $t$, and then normalized with respect
            to the maximum timestamp in each batch.
            Additional transforms to the temporal feature can be added to the
            ``extra_transforms`` parameter, with the `__time__` key.
            The "relative_legacy" format is provided for old checkpoints where the time
            transform was using :class:`StandardScaler` for normalization instead of
            :class:`MaxScaler`.
        noise_std : float
            The standard deviation of the noise to add to the input data. This is useful
            for adding noise to the input data to make the model more robust to noise in
            the data.
        """
        super().__init__(
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
            known_covariates=known_covariates,
            target_covariate=target_covariate,
            known_past_covariates=known_past_covariates,
            normalize=normalize,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            extra_transforms=extra_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
            shuffle=shuffle,
            distributed=distributed,
        )

        self.save_hyperparameters(ignore=["extra_transforms"])

        self.hparams["known_covariates"] = _to_list(self.hparams["known_covariates"])
        self.hparams["known_past_covariates"] = (
            _to_list(self.hparams["known_past_covariates"])
            if self.hparams["known_past_covariates"] is not None
            else []
        )

    def _create_transforms(self) -> None:
        super()._create_transforms()
        if self.hparams["time_column"] is None:
            return

        if self.hparams["normalize"]:
            if self.hparams["time_format"] == "relative":
                transforms = [
                    DeltaTransform(),
                    MaxScaler(num_features_=1),
                ]
            elif self.hparams["time_format"] == "relative_legacy":
                transforms = [
                    DeltaTransform(),
                    StandardScaler(num_features_=1),
                ]
            elif self.hparams["time_format"] == "absolute":
                transforms = [MaxScaler(num_features_=1)]
            else:
                msg = (
                    f"Unknown time format {self.hparams['time_format']}. "
                    "Expected 'relative' or 'absolute'."
                )
                raise ValueError(msg)
        else:
            transforms = []

        if TIME in self._extra_transforms_source:
            transforms += typing.cast(
                list[BaseTransform], self._extra_transforms_source[TIME]
            )

        self._transforms[TIME] = TransformCollection(*transforms)

    def _fit_transforms(self, dfs: list[pd.DataFrame]) -> None:
        super()._fit_transforms(dfs)

        if self.hparams["time_column"] is not None:
            if self.hparams["time_format"] in ["relative", "relative_legacy"]:
                self._fit_relative_time(
                    dfs, self._transforms[TIME], stride=self.hparams["stride"]
                )
            elif self.hparams["time_format"] == "absolute":
                self._fit_absolute_time(dfs, self._transforms[TIME])

    @staticmethod
    def _fit_relative_time(
        dfs: list[pd.DataFrame], transform: BaseTransform, stride: int = 1
    ) -> None:
        """
        Stride is a form of downsamle, since samples are taken at `stride` intervals,
        so we need to fit the scalers taking this into account. The relative time
        uses dt (relative time difference), so we only need to find the largest
        dt possible to fit a MaxScaler (since dt is assumed to be always positive).
        """
        for df in dfs:
            for start in range(stride):
                time = df[TIME].iloc[start::stride].to_numpy(dtype=float)
                if isinstance(transform, TransformCollection):
                    # apply the transforms iteratively. Later transforms must be fitted
                    # on already transformerd data. For the delta transform, we need to
                    # shave off the first element, which is set to zero by the transform.
                    for t in transform:
                        time = t.fit_transform(time)
                        if isinstance(t, DeltaTransform):
                            time = time[1:]
                else:
                    transform.fit(time)

    def _fit_absolute_time(
        self, dfs: list[pd.DataFrame], transform: BaseTransform
    ) -> None:
        """
        Fits the absolute time scaler to the data. Since absolute time is monotonically
        increasing, the scaling transform must be fitted on samples directly, so we
        create sliding window samples to fit the largest time axis possible.
        """
        wgs = [
            WindowGenerator(
                num_points=len(df),
                window_size=self.hparams["ctxt_seq_len"] + self.hparams["tgt_seq_len"],
                stride=1,
                zero_pad=False,
            )
            for df in dfs
        ]

        dts = []
        for df, wg in zip(dfs, wgs, strict=True):
            for start in range(self.hparams["stride"]):
                time = df[TIME].to_numpy(dtype=float)[start :: self.hparams["stride"]]

                dt = _calc_fast_absolute_dt(time, numba.typed.List(wg))
                dts.append(dt)

        transform.fit(np.concatenate(dts).flatten())

    @property
    def ctxt_seq_len(self) -> int:
        """Exposes context sequence length for LightningCLI"""
        return self.hparams["ctxt_seq_len"]

    @property
    def tgt_seq_len(self) -> int:
        """Exposes context sequence length for LightningCLI"""
        return self.hparams["tgt_seq_len"]


@numba.njit
def _calc_fast_absolute_dt(time: np.ndarray, slices: list[slice]) -> np.ndarray:
    """
    Calculates the absolute time difference between the start of the slice and the
    rest of the slice. This is a fast implementation using Numba.
    """
    dt = np.zeros(len(slices))
    for i, sl in enumerate(slices):
        dt[i] = np.max(np.abs(time[sl] - time[sl.start]))
    return dt


class EncoderDecoderDataModule(TransformerDataModule):
    @override
    def _make_dataset_from_df(
        self,
        df: pd.DataFrame | list[pd.DataFrame],
        *,
        predict: bool = False,
    ) -> EncoderDecoderDataset:
        input_cols = [cov.col for cov in self.known_covariates]
        known_past_cols = [cov.col for cov in self.known_past_covariates]

        time_format: typing.Literal["relative", "absolute"] = (
            "relative"
            if self.hparams["time_format"] in {"relative", "relative_legacy"}
            else "absolute"
        )

        return EncoderDecoderDataset(
            input_data=df[input_cols]
            if isinstance(df, pd.DataFrame)
            else [df[input_cols] for df in df],
            known_past_data=df[known_past_cols]
            if isinstance(df, pd.DataFrame)
            else [df[known_past_cols] for df in df]
            if len(known_past_cols) > 0
            else None,
            target_data=df[self.target_covariate.col]  # type: ignore[arg-type]
            if isinstance(df, pd.DataFrame)
            else [df[self.target_covariate.col] for df in df],
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            time_data=(  # type: ignore[arg-type]
                df[TIME] if isinstance(df, pd.DataFrame) else [df[TIME] for df in df]
            )
            if self.hparams["time_column"] is not None
            else None,
            time_format=time_format,
            stride=self.hparams["stride"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            transforms=self.transforms,
            noise_std=self.hparams["noise_std"] if not predict else 0.0,
            dtype=self.hparams["dtype"],
            add_target_to_past=self.hparams["add_target_to_past"],
        )

    @staticmethod
    def collate_fn(  # type: ignore[override]
        samples: list[EncoderDecoderTargetSample],
    ) -> EncoderDecoderTargetSample:
        """
        Cuts samples so that the length of the sequence (dimension 1) is the shortest possible
        if randomize_seq_len is True, which causes sequences to be randomly cut to a shorter
        length than the maximum by the dataset classs.

        Encoder sequences are cut from the beginning of the sequence, and decoder sequences
        are cut from the end. The sequence length is determined by the longest sequence in
        the samples.
        """
        if all("encoder_lengths" in sample for sample in samples):
            max_enc_len = max(sample["encoder_lengths"] for sample in samples)
        else:
            max_enc_len = samples[0]["encoder_input"].size(1)

        max_enc_len = int(max_enc_len)

        if all("decoder_lengths" in sample for sample in samples):
            max_tgt_len = max(sample["decoder_lengths"] for sample in samples)
        else:
            max_tgt_len = samples[0]["decoder_input"].size(1)

        assert max_tgt_len > 0

        max_tgt_len = int(max_tgt_len)

        cut_samples = []
        for sample in samples:
            cut_sample = {
                "encoder_input": sample["encoder_input"][-max_enc_len:],
                "decoder_input": sample["decoder_input"][:max_tgt_len],
                "target": sample["target"][:max_tgt_len],
            }
            if "encoder_lengths" in sample:
                cut_sample["encoder_lengths"] = sample["encoder_lengths"]
            if "decoder_lengths" in sample:
                cut_sample["decoder_lengths"] = sample["decoder_lengths"]

            if "encoder_mask" in sample:
                cut_sample["encoder_mask"] = sample["encoder_mask"][-max_enc_len:]

            if "decoder_mask" in sample:
                cut_sample["decoder_mask"] = sample["decoder_mask"][:max_tgt_len]

            cut_samples.append(cut_sample)

        return typing.cast(
            EncoderDecoderTargetSample[torch.Tensor],
            torch.utils.data.dataloader.default_collate(cut_samples),
        )


class EncoderDataModule(TransformerDataModule):
    @override
    def _make_dataset_from_df(
        self,
        df: pd.DataFrame | list[pd.DataFrame],
        *,
        predict: bool = False,
    ) -> EncoderDataset:
        input_cols = [cov.col for cov in self.known_covariates]

        return EncoderDataset(
            input_data=df[input_cols]
            if isinstance(df, pd.DataFrame)
            else [df[input_cols] for df in df],
            target_data=df[self.target_covariate.col]  # type: ignore[arg-type]
            if isinstance(df, pd.DataFrame)
            else [df[self.target_covariate.col] for df in df],
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            stride=self.hparams["stride"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            transforms=self.transforms,
            dtype=self.hparams["dtype"],
        )
