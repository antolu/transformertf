from __future__ import annotations

import typing

import pandas as pd
import pytest

from transformertf.data import EncoderDecoderDataModule


@pytest.fixture(scope="module")
def transformer_datamodule_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "known_covariates": [current_key],
        "target_covariate": field_key,
        "train_df_paths": df_path,
        "val_df_paths": df_path,
        "normalize": True,
        "ctxt_seq_len": 200,
        "tgt_seq_len": 100,
        "randomize_seq_len": True,
        "min_ctxt_seq_len": 10,
        "min_tgt_seq_len": 10,
        "stride": 1,
        "downsample": 1,
        "downsample_method": "interval",
        "target_depends_on": None,
        "extra_transforms": None,
        "batch_size": 16,
        "num_workers": 0,
        "dtype": "float32",
        "distributed": False,
    }


@pytest.fixture(scope="module")
def datamodule_transformer(
    transformer_datamodule_config: dict[str, typing.Any],
) -> EncoderDecoderDataModule:
    dm = EncoderDecoderDataModule(**transformer_datamodule_config)
    dm.prepare_data()
    dm.setup()

    return dm


def test_sample_length(
    datamodule_transformer: EncoderDecoderDataModule,
) -> None:
    dataloader = datamodule_transformer.train_dataloader()

    batch = next(iter(dataloader))

    encoder_input = batch["encoder_input"]

    assert max(batch["encoder_lengths"]) <= 200
    assert min(batch["encoder_lengths"]) >= 10

    assert max(batch["decoder_lengths"]) <= 100
    assert min(batch["decoder_lengths"]) >= 10

    assert max(batch["decoder_lengths"]) == batch["decoder_input"].shape[1]
    assert max(batch["decoder_lengths"]) == batch["target"].shape[1]

    assert max(batch["encoder_lengths"]) == encoder_input.shape[1]

    assert encoder_input.shape[1] >= 10


def test_sample_length_distribution(
    datamodule_transformer: EncoderDecoderDataModule,
) -> None:
    dataloader = datamodule_transformer.train_dataloader()

    encoder_lens = []
    decoder_lens = []

    for i, batch in enumerate(dataloader):
        if i > 100:
            break
        encoder_lens.append(batch["encoder_lengths"].max())
        decoder_lens.append(batch["decoder_lengths"].max())

    assert pd.Series(encoder_lens).mean() > 10
    assert pd.Series(encoder_lens).mean() < 200

    assert pd.Series(decoder_lens).mean() > 10
    assert pd.Series(decoder_lens).mean() < 100
