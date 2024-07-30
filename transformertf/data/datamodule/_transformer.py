from __future__ import annotations

import sys
import typing

import pandas as pd

from .._downsample import DOWNSAMPLE_METHODS
from ..dataset import EncoderDataset, EncoderDecoderDataset
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
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
        shuffle: bool = True,
        distributed_sampler: bool = False,
    ):
        """
        For documentation of arguments see :class:`DataModuleBase`.
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
            distributed_sampler=distributed_sampler,
        )

        self.save_hyperparameters(ignore=["extra_transforms"])

        self.hparams["known_covariates"] = _to_list(self.hparams["known_covariates"])
        self.hparams["known_past_covariates"] = (
            _to_list(self.hparams["known_past_covariates"])
            if self.hparams["known_past_covariates"] is not None
            else []
        )

    @property
    def ctxt_seq_len(self) -> int:
        return self.hparams["ctxt_seq_len"]

    @property
    def tgt_seq_len(self) -> int:
        return self.hparams["tgt_seq_len"]


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

        return EncoderDecoderDataset(
            input_data=df[input_cols]
            if isinstance(df, pd.DataFrame)
            else [df[input_cols] for df in df],
            known_past_data=df[known_past_cols]
            if isinstance(df, pd.DataFrame)
            else [df[known_past_cols] for df in df]
            if len(known_past_cols) > 0
            else None,
            target_data=df[self.target_covariate.col]
            if isinstance(df, pd.DataFrame)
            else [df[self.target_covariate.col] for df in df],
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            stride=self.hparams["stride"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            input_transforms=self.input_transforms,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
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
            target_data=df[self.target_covariate.col]
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
            input_transforms=self.input_transforms,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
        )
