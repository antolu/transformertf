from __future__ import annotations

import pandas as pd
import pytest

from transformertf.models.phylstm import (
    PhyLSTM,
    PhyLSTMDataModule,
)
from transformertf.utils.predict import predict_phylstm


@pytest.fixture()
def past_covariates(df: pd.DataFrame) -> pd.DataFrame:
    return df[["I_meas_A"]].iloc[:1000]


@pytest.fixture()
def future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    return df[["I_meas_A"]].iloc[1000:10000]


@pytest.fixture(scope="module")
def phylstm_module() -> PhyLSTM:
    return PhyLSTM(sequence_length=100)


@pytest.fixture(scope="module")
def phylstm_datamodule(
    df_path: str,
    current_key: str,
    field_key: str,
) -> PhyLSTMDataModule:
    dm = PhyLSTMDataModule(
        input_columns=current_key,
        target_column=field_key,
        seq_len=100,
        train_df=df_path,
        val_df=df_path,
    )
    dm.prepare_data()
    return dm


def test_predict_phylstm(
    phylstm_module: PhyLSTM,
    phylstm_datamodule: PhyLSTMDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
) -> None:
    predict_phylstm(
        module=phylstm_module,
        datamodule=phylstm_datamodule,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )
