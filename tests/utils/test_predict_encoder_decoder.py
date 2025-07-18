"""
Test the predict_transformer function.

This should probably be an integration test, but it's here for now.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transformertf.data.datamodule import EncoderDecoderDataModule
from transformertf.models.transformer import (
    VanillaTransformer,
)


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
def encoder_decoder_module() -> VanillaTransformer:
    return VanillaTransformer(
        num_features=2,
        ctxt_seq_len=100,
        tgt_seq_len=30,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=4,
        n_dim_model=32,
        fc_dim=64,
    )


@pytest.fixture(scope="module")
def encoder_decoder_datamodule(
    df_path: str,
    current_key: str,
    field_key: str,
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(
        known_covariates=[current_key],
        target_covariate=field_key,
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        ctxt_seq_len=100,
        tgt_seq_len=30,
    )


# def test_predict_encoder_decoder(
#     encoder_decoder_module: VanillaTransformer,
#     encoder_decoder_datamodule: EncoderDecoderDataModule,
#     past_covariates: pd.DataFrame,
#     future_covariates: pd.DataFrame,
#     past_target: pd.DataFrame,
# ) -> None:
#     predict_encoder_decoder(
#         encoder_decoder_module,
#         encoder_decoder_datamodule,
#         past_covariates,
#         future_covariates,
#         past_target,
#     )
