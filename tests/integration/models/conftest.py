from __future__ import annotations

import typing

import pytest

from transformertf.data import EncoderDecoderDataModule


@pytest.fixture(scope="module")
def encoder_decoder_datamodule_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "train_df": df_path,
        "val_df": df_path,
        "num_workers": 0,
        "target_depends_on": current_key,
        "input_columns": [current_key],
        "target_column": field_key,
        "ctxt_seq_len": 100,
        "tgt_seq_len": 50,
    }


@pytest.fixture()
def encoder_decoder_datamodule(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(**encoder_decoder_datamodule_config)
