from __future__ import annotations

import typing

from transformertf.data.dataset import EncoderDataset, EncoderDecoderDataset

from ._base import DataModuleBase

if typing.TYPE_CHECKING:
    import numpy as np

    from ..transform import BaseTransform


class TransformerDataModule(DataModuleBase):
    def __init__(
        self,
        *,
        input_columns: str | typing.Sequence[str],
        target_column: str,
        known_past_columns: str | typing.Sequence[str] | None = None,
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
        downsample_method: typing.Literal[
            "interval", "average", "convolve"
        ] = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
        distributed_sampler: bool = False,
    ):
        super().__init__(
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
            input_columns=input_columns,
            target_column=target_column,
            known_past_columns=known_past_columns,
            normalize=normalize,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
            distributed_sampler=distributed_sampler,
        )

        self.save_hyperparameters()

    @property
    def ctxt_seq_len(self) -> int:
        return self.hparams["ctxt_seq_len"]

    @property
    def tgt_seq_len(self) -> int:
        return self.hparams["tgt_seq_len"]


class EncoderDecoderDataModule(TransformerDataModule):
    def _make_dataset_from_arrays(
        self,
        input_data: np.ndarray,
        known_past_data: np.ndarray | None = None,
        target_data: np.ndarray | None = None,
        *,
        predict: bool = False,
    ) -> EncoderDecoderDataset:
        if target_data is None:
            msg = "Target data must be provided for an encoder-decoder model."
            raise ValueError(msg)

        return EncoderDecoderDataset(
            input_data=input_data,
            known_past_data=known_past_data,
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
        known_past_data: np.ndarray | None = None,
        target_data: np.ndarray | None = None,
        *,
        predict: bool = False,
    ) -> EncoderDataset:
        if target_data is None:
            msg = "Target data should not be provided for an encoder model."
            raise ValueError(msg)

        if known_past_data is not None:
            msg = "known_past_data is not used in this class."
            raise NotImplementedError(msg)

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
