from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from transformertf.data import EncoderDecoderDataModule


@pytest.fixture(scope="module")
def encoder_decoder_datamodule_config(
    df_path: str, current_key: str, field_key: str, tmp_path_factory: typing.Any
) -> dict[str, typing.Any]:
    # copy the df to the tempdir
    df = pd.read_parquet(df_path)
    df[f"{current_key}_dot"] = np.gradient(df[current_key])

    df_path = tmp_path_factory.mktemp("data").joinpath("df.parquet")

    df.to_parquet(df_path)

    return {
        "train_df_paths": df_path,
        "val_df_paths": df_path,
        "num_workers": 0,
        "known_covariates": [f"{current_key}_dot", current_key],
        "target_covariate": field_key,
        "time_column": "time_ms",
        "time_format": "absolute",
        "ctxt_seq_len": 100,
        "tgt_seq_len": 50,
        "batch_size": 32,
    }


@pytest.fixture
def encoder_decoder_datamodule(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(**encoder_decoder_datamodule_config)
