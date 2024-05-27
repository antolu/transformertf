"""
Test that the transformer datamodule produces the correct dataset samples
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transformertf.data import EncoderDecoderDataModule

DF_PATH = str(Path(__file__).parent.parent.parent / "sample_data.parquet")
CURRENT = "I_meas_A"
FIELD = "B_meas_T"


@pytest.fixture(scope="module")
def datamodule_transformer(
    df_path: str, current_key: str, field_key: str
) -> EncoderDecoderDataModule:
    dm = EncoderDecoderDataModule(
        train_df_paths=df_path,
        val_df_paths=df_path,
        known_covariates=[current_key],
        target_covariate=field_key,
        known_past_covariates=["time_ms"],
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_correct_dataset_sample(
    datamodule_transformer: EncoderDecoderDataModule,
) -> None:
    dataset = datamodule_transformer.train_dataset

    sample = dataset[0]

    assert sample["encoder_input"].shape[-1] == 3
    assert sample["decoder_input"].shape[-1] == 3
    print()
