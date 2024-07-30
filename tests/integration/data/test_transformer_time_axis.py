from __future__ import annotations

import pytest

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
def relative_time_encoder_decoder_datamodule(
    df_path: str, current_key: str, field_key: str, time_key: str
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        known_covariates=[current_key],
        target_covariate=field_key,
        time_column=time_key,
        time_format="relative",
    )


def test_relative_time_encoder_decoder_datamodule_setup(
    relative_time_encoder_decoder_datamodule: EncoderDecoderDataModule,
) -> None:
    relative_time_encoder_decoder_datamodule.prepare_data()
    relative_time_encoder_decoder_datamodule.setup()


def test_relative_time_encoder_decoder_datamodule_transforms(
    relative_time_encoder_decoder_datamodule: EncoderDecoderDataModule,
) -> None:
    assert TIME_PREFIX in relative_time_encoder_decoder_datamodule.input_transforms


@pytest.fixture
def absolute_time_encoder_decoder_datamodule(
    df_path: str, current_key: str, field_key: str, time_key: str
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule(
        train_df_paths=[df_path],
        val_df_paths=[df_path],
        known_covariates=[current_key],
        target_covariate=field_key,
        time_column=time_key,
        time_format="absolute",
    )


def test_absolute_time_encoder_decoder_datamodule_setup(
    absolute_time_encoder_decoder_datamodule: EncoderDecoderDataModule,
) -> None:
    absolute_time_encoder_decoder_datamodule.prepare_data()
    absolute_time_encoder_decoder_datamodule.setup()


def test_absolute_time_encoder_decoder_datamodule_transforms(
    absolute_time_encoder_decoder_datamodule: EncoderDecoderDataModule,
) -> None:
    assert TIME_PREFIX in absolute_time_encoder_decoder_datamodule.input_transforms
