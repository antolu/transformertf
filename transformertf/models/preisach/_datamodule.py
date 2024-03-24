from __future__ import annotations

import logging
import typing
import warnings

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
        train_df: pd.DataFrame | list[pd.DataFrame],
        val_df: pd.DataFrame | list[pd.DataFrame],
        known_covariates_cols: str | typing.Sequence[str] = (CURRENT,),
        target_col: str = FIELD,
        lowpass_filter: bool = False,
        mean_filter: bool = False,
        downsample: int = 1,
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        num_workers: int = 0,
    ):
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            known_covariates_cols=known_covariates_cols,
            target_col=target_col,
            normalize=False,
            downsample=downsample,
            target_depends_on=target_depends_on or known_covariates_cols[0],
            extra_transforms=extra_transforms,
            batch_size=1,
            num_workers=num_workers,
            dtype="float64",
        )
        super().save_hyperparameters(
            ignore=["train_df", "val_df", "known_covariates_cols"]
        )

    @classmethod
    def parse_config_kwargs(
        cls, config: PreisachConfig, **kwargs: typing.Any  # type: ignore[override]
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        default_kwargs = {}
        default_kwargs.update(kwargs)

        # remove past and static covariates
        for key in [
            "past_covariates_cols",
            "static_cont_covariates_cols",
            "static_cat_covariates_cols",
        ]:
            if key in default_kwargs:
                warnings.warn(
                    f"{key} is not used in PreisachDataModule, ignoring it"
                )
                default_kwargs.pop(key, None)

        return default_kwargs
