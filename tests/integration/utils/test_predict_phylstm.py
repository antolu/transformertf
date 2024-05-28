from __future__ import annotations

import pandas as pd
import pytest

from transformertf.data import TimeSeriesDataModule
from transformertf.models.bwlstm import (
    BoucWenLSTM,
)
from transformertf.utils.predict import predict_phylstm


@pytest.fixture()
def past_covariates(df: pd.DataFrame) -> pd.DataFrame:
    return df[["I_meas_A"]].iloc[:1000]


@pytest.fixture()
def future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    return df[["I_meas_A"]].iloc[1000:10000]


@pytest.fixture(scope="module")
def phylstm_module() -> BoucWenLSTM:
    return BoucWenLSTM(seq_len=100)


@pytest.fixture(scope="module")
def phylstm_datamodule(
    df_path: str,
    current_key: str,
    field_key: str,
) -> TimeSeriesDataModule:
    dm = TimeSeriesDataModule(
        known_covariates=current_key,
        target_covariate=field_key,
        seq_len=100,
        train_df_paths=df_path,
        val_df_paths=df_path,
    )
    dm.prepare_data()
    return dm


def test_predict_phylstm(
    phylstm_module: BoucWenLSTM,
    phylstm_datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
) -> None:
    predict_phylstm(
        module=phylstm_module,
        datamodule=phylstm_datamodule,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )
