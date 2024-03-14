"""
Test the predict_transformer function.

This should probably be an integration test, but it's here for now.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transformertf.data.datamodule import EncoderDecoderDataModule
from transformertf.data.transform import DivideByXTransform
from transformertf.models.transformer import (
    VanillaTransformerConfig,
    VanillaTransformerModule,
)
from transformertf.utils.predict import predict_encoder_decoder


@pytest.fixture
def past_covariates(df: pd.DataFrame) -> pd.DataFrame:
    return df[["I_meas_A"]].iloc[:1000]


@pytest.fixture
def future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    return df[["I_meas_A"]].iloc[1000:10000]


@pytest.fixture
def past_target(df: pd.DataFrame) -> pd.DataFrame:
    return df[["B_meas_T"]].iloc[:1000]


@pytest.fixture(scope="module")
def encoder_decoder_config() -> VanillaTransformerConfig:
    return VanillaTransformerConfig(
        n_dim_model=16,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        fc_dim=32,
        dropout=0.0,
        ctxt_seq_len=100,
        tgt_seq_len=30,
        batch_size=8,
        normalize=True,
        downsample=5,
        downsample_method="average",
        extra_transforms={"B_meas_T": [DivideByXTransform()]},
        target_depends_on="I_meas_A",
        input_columns=["I_meas_A"],
        target_column="B_meas_T",
    )


@pytest.fixture(scope="module")
def encoder_decoder_module(
    encoder_decoder_config: VanillaTransformerConfig, df: pd.DataFrame
) -> VanillaTransformerModule:
    return VanillaTransformerModule.from_config(encoder_decoder_config)


@pytest.fixture(scope="module")
def encoder_decoder_datamodule(
    df: pd.DataFrame,
    encoder_decoder_config: VanillaTransformerConfig,
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule.from_dataframe(
        encoder_decoder_config,
        train_df=df,
        val_df=df,
    )


def test_predict_encoder_decoder(
    encoder_decoder_module: VanillaTransformerModule,
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
) -> None:
    predict_encoder_decoder(
        encoder_decoder_module,
        encoder_decoder_datamodule,
        past_covariates,
        future_covariates,
        past_target,
    )
