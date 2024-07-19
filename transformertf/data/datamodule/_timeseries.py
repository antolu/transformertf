from __future__ import annotations

import typing

from transformertf.data.dataset import TimeSeriesDataset

from ._base import DataModuleBase

if typing.TYPE_CHECKING:
    import numpy as np

    from .._downsample import DOWNSAMPLE_METHODS
    from ..transform import BaseTransform


class TimeSeriesDataModule(DataModuleBase):
    """
    Specfic datamodule for time series data, where
    the models map a sequence of input covariates to a target covariate,
    i.e. I: [bs, seq_len, n_covariates] -> T: [bs, seq_len, 1].
    """

    def __init__(
        self,
        *,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        normalize: bool = True,
        seq_len: int = 200,
        min_seq_len: int | None = None,
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
            known_past_covariates=None,
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

        self.hparams["known_covariates"] = self._to_list(
            self.hparams["known_covariates"]
        )

    def _make_dataset_from_arrays(
        self,
        input_data: np.ndarray,
        known_past_data: np.ndarray | None = None,
        target_data: np.ndarray | None = None,
        *,
        predict: bool = False,
    ) -> TimeSeriesDataset:
        if known_past_data is not None and known_past_data.size > 0:
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

    @property
    def seq_len(self) -> int:
        """
        Returns the sample sequence length. This is used by LightningCLI
        to link arguments to the model.

        Returns
        -------
        int
            Sample sequence length
        """
        return self.hparams["seq_len"]

    @property
    def num_past_known_covariates(self) -> int:
        """
        Returns the number of past known covariates. This is used by LightningCLI
        to link arguments to the model.

        Returns
        -------
        int
            Number of past known covariates
        """
        return len(self.hparams["known_covariates"])
