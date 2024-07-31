from __future__ import annotations

import pytest
import torch

from transformertf.data import EncoderDecoderDataModule, TransformerDataModule
from transformertf.data.datamodule._base import TIME_PREFIX  # noqa: PLC2701


def test_transformer_datamodule_create_with_time_axis(
    df_path: str, current_key: str, field_key: str
) -> None:
    dm = TransformerDataModule(
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        known_covariates=[current_key],
        target_covariate=field_key,
        time_column="time_ms",
    )

    assert dm is not None


@pytest.fixture
def relative_time_datamodule(
    df_path: str, current_key: str, field_key: str, time_key: str
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        known_covariates=[current_key],
        target_covariate=field_key,
        time_column=time_key,
        time_format="relative",
        downsample=20,
    )


def test_relative_time_encoder_decoder_datamodule_setup(
    relative_time_datamodule: EncoderDecoderDataModule,
) -> None:
    relative_time_datamodule.prepare_data()
    relative_time_datamodule.setup()


def test_relative_time_dataset(
    relative_time_datamodule: EncoderDecoderDataModule,
) -> None:
    relative_time_datamodule.prepare_data()
    relative_time_datamodule.setup()

    dataset = relative_time_datamodule.train_dataset
    sample = dataset[0]

    assert sample["encoder_input"].shape[-1] == 3  # (time, current, field)
    assert sample["encoder_input"][0, 0] == 0.0  # time starts at 0

    std = torch.std(sample["encoder_input"][:, 0])
    assert std <= 1.0, f"Standard deviation of time is {std}"


def test_relative_time_encoder_decoder_datamodule_transforms(
    relative_time_datamodule: EncoderDecoderDataModule,
) -> None:
    assert TIME_PREFIX in relative_time_datamodule.transforms


@pytest.fixture
def absolute_time_datamodule(
    df_path: str, current_key: str, field_key: str, time_key: str
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        known_covariates=[current_key],
        target_covariate=field_key,
        time_column=time_key,
        time_format="absolute",
        downsample=20,
    )


def test_absolute_time_encoder_decoder_datamodule_setup(
    absolute_time_datamodule: EncoderDecoderDataModule,
) -> None:
    absolute_time_datamodule.prepare_data()
    absolute_time_datamodule.setup()


def test_absolute_time_encoder_decoder_datamodule_transforms(
    absolute_time_datamodule: EncoderDecoderDataModule,
) -> None:
    assert TIME_PREFIX in absolute_time_datamodule.transforms


def test_absolute_time_dataset(
    absolute_time_datamodule: EncoderDecoderDataModule,
) -> None:
    absolute_time_datamodule.prepare_data()
    absolute_time_datamodule.setup()

    dataset = absolute_time_datamodule.train_dataset
    sample = dataset[0]

    assert sample["encoder_input"].shape[-1] == 3  # (time, current, field)
    assert sample["encoder_input"][0, 0] == 0.0  # time starts at 0
    assert (torch.diff(sample["encoder_input"][:, 0]) > 0.0).all()  # time is increasing
    assert torch.max(sample["encoder_input"][:, 0]) <= 1.0  # time is normalized

    assert sample is not None


def test_absolute_time_dataset_random(
    df_path: str, current_key: str, field_key: str, time_key: str
) -> None:
    datamodule = EncoderDecoderDataModule(
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        known_covariates=[current_key],
        target_covariate=field_key,
        time_column=time_key,
        time_format="absolute",
        downsample=20,
        randomize_seq_len=True,
        min_ctxt_seq_len=250,
        min_tgt_seq_len=300,
    )
    datamodule.prepare_data()
    datamodule.setup()

    dataset = datamodule.train_dataset
    sample = dataset[0]

    assert sample["encoder_input"].shape[-1] == 3  # (time, current, field)
    encoder_lengths = (
        (sample["encoder_lengths"].item() + 1.0) * datamodule.ctxt_seq_len / 2
    )
    seq_start = int(datamodule.hparams["ctxt_seq_len"] - encoder_lengths)

    assert sample["encoder_input"][seq_start, 0] == 0.0  # time starts at 0
    assert (sample["encoder_input"][:seq_start, 0] == 0.0).all()  # time is zero-padded
    assert (sample["encoder_input"][seq_start:, 0] >= 0.0).all()  # time is increasing
    assert torch.max(sample["encoder_input"][:, 0]) <= 1.0  # time is normalized
