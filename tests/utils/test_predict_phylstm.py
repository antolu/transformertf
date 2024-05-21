from __future__ import annotations

import pandas as pd
import pytest

from transformertf.data.transform import DivideByXTransform
from transformertf.models.phylstm import (
    PhyLSTM,
    PhyLSTMConfig,
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
def phylstm_config() -> PhyLSTMConfig:
    return PhyLSTMConfig(
        num_layers=1,
        hidden_size=64,
        hidden_size_fc=64,
        seq_len=100,
        downsample=5,
        dropout=0.0,
        batch_size=8,
        normalize=True,
        downsample_method="average",
        extra_transforms={"B_meas_T": [DivideByXTransform()]},
        target_depends_on="I_meas_A",
        input_columns="I_meas_A",
        target_column="B_meas_T",
    )


@pytest.fixture(scope="module")
def phylstm_module(phylstm_config: PhyLSTMConfig, df: pd.DataFrame) -> PhyLSTM:
    return PhyLSTM.from_config(phylstm_config)


@pytest.fixture(scope="module")
def phylstm_datamodule(
    df: pd.DataFrame,
    phylstm_config: PhyLSTMConfig,
) -> PhyLSTMDataModule:
    dm = PhyLSTMDataModule.from_dataframe(
        phylstm_config,
        train_df=df,
        val_df=df,
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
