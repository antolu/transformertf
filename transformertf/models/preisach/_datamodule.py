from __future__ import annotations

import logging
import typing

from ...data import DataModuleBase
from ._config import PreisachConfig

log = logging.getLogger(__name__)


CURRENT = "I_meas_A"
FIELD = "B_meas_T"


class PreisachDataModule(DataModuleBase):
    TRANSFORMS = ["polynomial", "normalize"]

    def __init__(
        self,
        train_dataset: str | typing.Sequence[str] | None = None,
        val_dataset: str | typing.Sequence[str] | None = None,
        lowpass_filter: bool = False,
        mean_filter: bool = False,
        downsample: int = 1,
        remove_polynomial: bool = True,
        polynomial_degree: int = 1,
        polynomial_iterations: int = 1000,
        num_workers: int = 0,
        current_column: str = CURRENT,
        field_column: str = FIELD,
        model_dir: str | None = None,
    ):
        super().__init__(
            input_columns=[current_column],
            target_columns=[field_column],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            normalize=True,
            seq_len=None,
            randomize_seq_len=False,
            min_seq_len=None,
            batch_size=1,
            downsample=downsample,
            remove_polynomial=remove_polynomial,
            polynomial_degree=polynomial_degree,
            polynomial_iterations=polynomial_iterations,
            num_workers=num_workers,
        )
        super().save_hyperparameters(ignore=["current_column", "field_column"])

    @classmethod
    def from_config(cls, config: PreisachConfig, **kwargs: typing.Any) -> PreisachDataModule:  # type: ignore[override]
        default_kwargs = {
            "train_dataset": config.train_dataset,
            "val_dataset": config.val_dataset,
            "seq_len": config.seq_len,
            "downsample": config.downsample,
            "num_workers": config.num_workers,
            "model_dir": config.model_dir,
            "lowpass_filter": config.lowpass_filter,
            "mean_filter": config.mean_filter,
            "remove_polynomial": config.remove_polynomial,
            "polynomial_degree": config.polynomial_degree,
            "polynomial_iterations": config.polynomial_iterations,
            "current_column": config.current_column,
            "field_column": config.field_column,
        }
        default_kwargs.update(kwargs)

        return cls(**default_kwargs) # type: ignore[call-arg,arg-type]
