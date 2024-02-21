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
        input_columns: str | typing.Sequence[str],
        target_column: str,
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
        remove_polynomial: bool = False,
        polynomial_degree: int = 1,
        polynomial_iterations: int = 1000,
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
    ):
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            input_columns=input_columns,
            target_column=target_column,
            normalize=normalize,
            downsample=downsample,
            downsample_method=downsample_method,
            remove_polynomial=remove_polynomial,
            polynomial_degree=polynomial_degree,
            polynomial_iterations=polynomial_iterations,
            target_depends_on=target_depends_on,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
        )

        self.save_hyperparameters(ignore=["train_df", "val_df"])

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
        input_data: np.ndarray,
        target_data: np.ndarray | None = None,
        predict: bool = False,
    ) -> EncoderDecoderDataset:
        if target_data is None:
            raise ValueError(
                "Target data must be provided for an encoder-decoder model."
            )

        return EncoderDecoderDataset(
            input_data=input_data,
            target_data=target_data,
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
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
        input_data: np.ndarray,
        target_data: np.ndarray | None = None,
        predict: bool = False,
    ) -> EncoderDataset:
        if target_data is None:
            raise ValueError(
                "Target data should not be provided for an encoder model."
            )

        return EncoderDataset(
            input_data=input_data,
            target_data=target_data,
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            min_ctxt_seq_len=self.hparams["min_ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            stride=self.hparams["stride"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            input_transform=self.input_transforms,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
        )
