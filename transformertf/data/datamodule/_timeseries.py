from __future__ import annotations
import typing
import torch


from ._base import _DataModuleBase
from .._dataset import TimeSeriesDataset
from ...config import TimeSeriesBaseConfig

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np


class TimeSeriesDataModule(_DataModuleBase):
    def __init__(
            self,
            train_df: pd.DataFrame | list[pd.DataFrame],
            val_df: pd.DataFrame | list[pd.DataFrame],
            input_columns: str | typing.Sequence[str],
            target_column: str,
            normalize: bool = True,
            seq_len: int | None = None,
            min_seq_len: int | None = None,
            randomize_seq_len: bool = False,
            stride: int = 1,
            downsample: int = 1,
            remove_polynomial: bool = False,
            polynomial_degree: int = 1,
            polynomial_iterations: int = 1000,
            target_depends_on: str | None = None,
            batch_size: int = 128,
            num_workers: int = 0,
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            input_columns=input_columns,
            target_column=target_column,
            normalize=normalize,
            downsample=downsample,
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
            cls, config: TimeSeriesBaseConfig, **kwargs: typing.Any  # type: ignore[override]
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
            predict: bool = False,
    ) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            input_data=input_data,
            target_data=target_data,
            stride=self.hparams["stride"],
            seq_len=self.hparams["seq_len"],
            min_seq_len=self.hparams["min_seq_len"] if not predict else None,
            randomize_seq_len=self.hparams["randomize_seq_len"]
            if not predict
            else False,
            predict=predict,
            target_transform=self.target_transform,
            dtype=self.hparams["dtype"],
        )
