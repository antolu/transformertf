from __future__ import annotations

import logging
import typing

from ...data import TimeSeriesDataModule
from ._config import PreisachConfig

log = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    import pandas as pd

    from ...data import BaseTransform


CURRENT = "I_meas_A"
FIELD = "B_meas_T"


class PreisachDataModule(TimeSeriesDataModule):
    def __init__(
        self,
        input_columns: str | typing.Sequence[str] = (CURRENT,),
        target_column: str = FIELD,
        known_past_columns: str | typing.Sequence[str] | None = None,
        train_df: pd.DataFrame | list[pd.DataFrame] | None = None,
        val_df: pd.DataFrame | list[pd.DataFrame] | None = None,
        downsample: int = 1,
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        num_workers: int = 0,
    ):
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            input_columns=input_columns,
            target_column=target_column,
            known_past_columns=known_past_columns,
            normalize=False,
            downsample=downsample,
            target_depends_on=target_depends_on or input_columns[0],
            extra_transforms=extra_transforms,
            batch_size=1,
            num_workers=num_workers,
            dtype="float64",
        )
        super().save_hyperparameters(ignore=["train_df", "val_df"])

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: PreisachConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        default_kwargs = {}
        default_kwargs.update(kwargs)

        return default_kwargs
