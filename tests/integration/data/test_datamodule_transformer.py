"""
Test that the transformer datamodule produces the correct dataset samples
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transformertf.config import TransformerBaseConfig
from transformertf.data import EncoderDecoderDataModule

DF_PATH = str(Path(__file__).parent.parent.parent / "sample_data.parquet")
CURRENT = "I_meas_A"
FIELD = "B_meas_T"

config = TransformerBaseConfig()


@pytest.fixture(scope="module")
def datamodule_transformer() -> EncoderDecoderDataModule:
    dm = EncoderDecoderDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
        known_past_columns=["time_ms"],
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
