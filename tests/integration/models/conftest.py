from __future__ import annotations

import typing

import pytest

from transformertf.data import EncoderDecoderDataModule


@pytest.fixture(scope="module")
def encoder_decoder_datamodule_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "train_df_paths": df_path,
        "val_df_paths": df_path,
        "num_workers": 0,
        "known_covariates": [current_key],
        "target_covariate": field_key,
        "ctxt_seq_len": 100,
        "tgt_seq_len": 50,
    }


@pytest.fixture()
def encoder_decoder_datamodule(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(**encoder_decoder_datamodule_config)
