from __future__ import annotations

import typing

from transformertf.data.dataset import TimeSeriesDataset

from ._base import DataModuleBase

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from ...config import TimeSeriesBaseConfig
    from ..transform import BaseTransform


class TimeSeriesDataModule(DataModuleBase):
    def __init__(
        self,
        train_df: pd.DataFrame | list[pd.DataFrame] | None,
        val_df: pd.DataFrame | list[pd.DataFrame] | None,
        input_columns: str | typing.Sequence[str],
        target_column: str,
        normalize: bool = True,  # noqa: FBT001, FBT002
        seq_len: int | None = None,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,  # noqa: FBT001, FBT002
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
        *,
        distributed_sampler: bool = False,
    ):
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            input_columns=input_columns,
            target_column=target_column,
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

        self.save_hyperparameters(ignore=["train_df", "val_df"])

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: TimeSeriesBaseConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        default_kwargs = {
            "seq_len": config.seq_len,
            "min_seq_len": config.min_seq_len,
            "randomize_seq_len": config.randomize_seq_len,
            "stride": config.stride,
        }
        default_kwargs.update(kwargs)

        return default_kwargs

    def _make_dataset_from_arrays(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray | None = None,
        *,
        predict: bool = False,
    ) -> TimeSeriesDataset:
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
