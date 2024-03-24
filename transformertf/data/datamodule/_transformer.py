from __future__ import annotations

import typing

from transformertf.data.dataset import EncoderDataset, EncoderDecoderDataset
from ._base import DataModuleBase

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from ...config import TransformerBaseConfig
    from ..transform import BaseTransform


class TransformerDataModule(DataModuleBase):
    def __init__(
        self,
        train_df: pd.DataFrame | list[pd.DataFrame],
        val_df: pd.DataFrame | list[pd.DataFrame],
        known_covariates_cols: str | typing.Sequence[str] | None,
        target_col: str | None,
        past_covariates_cols: str | typing.Sequence[str] | None = None,
        static_cont_covariates_cols: str | typing.Sequence[str] | None = None,
        static_cat_covariates_cols: str | typing.Sequence[str] | None = None,
        normalize: bool = True,
        ctxt_seq_len: int = 500,
        tgt_seq_len: int = 300,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        downsample: int = 1,
        downsample_method: typing.Literal[
            "interval", "average", "convolve"
        ] = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
        input_columns: str | typing.Sequence[str] | None = None,  # deprecated
        target_column: str | None = None,  # deprecated
    ):
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            known_covariates_cols=known_covariates_cols,
            target_col=target_col,
            normalize=normalize,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
            input_columns=input_columns,
            target_column=target_column,
        )

        self.save_hyperparameters(
            ignore=[
                "train_df",
                "val_df",
                "known_covariates_cols",
                "past_covariates_cols",
                "static_cont_covariates_cols",
                "static_cat_covariates_cols",
                "input_columns",
                "target_column",
            ]
        )

    @classmethod
    def parse_config_kwargs(
        cls, config: TransformerBaseConfig, **kwargs: typing.Any  # type: ignore[override]
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        default_kwargs = {
            "ctxt_seq_len": config.ctxt_seq_len,
            "tgt_seq_len": config.tgt_seq_len,
            "min_ctxt_seq_len": config.min_ctxt_seq_len,
            "min_tgt_seq_len": config.min_tgt_seq_len,
            "randomize_seq_len": config.randomize_seq_len,
            "stride": config.stride,
        }
        default_kwargs.update(kwargs)

        return default_kwargs


class EncoderDecoderDataModule(TransformerDataModule):
    def _make_dataset_from_arrays(
        self,
        known_covariates_data: np.ndarray,
        target_data: np.ndarray | None = None,
        past_covariates_data: np.ndarray | None = None,
        static_cont_covariates_data: np.ndarray | None = None,
        static_cat_covariates_data: np.ndarray | None = None,
        predict: bool = False,
    ) -> EncoderDecoderDataset:
        if target_data is None:
            raise ValueError(
                "Target data must be provided for an encoder-decoder model."
            )

        return EncoderDecoderDataset(
            input_data=known_covariates_data,
            target_data=target_data,
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            past_covariates_data=past_covariates_data,
            static_cont_covariates_data=static_cont_covariates_data,
            static_cat_covariates_data=static_cat_covariates_data,
            stride=self.hparams["stride"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            input_transform=self.input_transforms,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
        )


class EncoderDataModule(TransformerDataModule):
    def _make_dataset_from_arrays(
        self,
        known_covariates_data: np.ndarray,
        target_data: np.ndarray | None = None,
        past_covariates_data: np.ndarray | None = None,
        static_cont_covariates_data: np.ndarray | None = None,
        static_cat_covariates_data: np.ndarray | None = None,
        predict: bool = False,
    ) -> EncoderDataset:
        if target_data is None:
            raise ValueError(
                "Target data should not be provided for an encoder model."
            )

        return EncoderDataset(
            input_data=known_covariates_data,
            target_data=target_data,
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            past_covariates_data=past_covariates_data,
            static_cont_covariates_data=static_cont_covariates_data,
            static_cat_covariates_data=static_cat_covariates_data,
            stride=self.hparams["stride"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            input_transform=self.input_transforms,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
        )
