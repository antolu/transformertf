from __future__ import annotations

import typing

from transformertf.data.dataset import TimeSeriesDataset

from ._base import DataModuleBase

if typing.TYPE_CHECKING:
    import numpy as np

    from ..transform import BaseTransform


class TimeSeriesDataModule(DataModuleBase):
    def __init__(
        self,
        *,
        input_columns: str | typing.Sequence[str],
        target_column: str,
        known_past_columns: str | typing.Sequence[str] | None = None,
        train_df: str | list[str] | None = None,
        val_df: str | list[str] | None = None,
        normalize: bool = True,
        seq_len: int = 200,
        min_seq_len: int | None = None,
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
            train_df=train_df,
            val_df=val_df,
            input_columns=input_columns,
            target_column=target_column,
            known_past_columns=known_past_columns,
            normalize=normalize,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            extra_transforms=extra_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
            distributed_sampler=distributed_sampler,
        )

        self.save_hyperparameters()

    def _make_dataset_from_arrays(
        self,
        input_data: np.ndarray,
        known_past_data: np.ndarray | None = None,
        target_data: np.ndarray | None = None,
        *,
        predict: bool = False,
    ) -> TimeSeriesDataset:
        if known_past_data is not None:
            msg = "known_past_data is not used in this class."
            raise NotImplementedError(msg)

        return TimeSeriesDataset(
            input_data=input_data,
            target_data=target_data,
            stride=self.hparams["stride"],
            seq_len=self.hparams["seq_len"],
            min_seq_len=self.hparams["min_seq_len"] if not predict else None,
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            predict=predict,
            input_transform=self.input_transforms,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
        )
