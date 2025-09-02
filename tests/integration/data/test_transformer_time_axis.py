from __future__ import annotations

import pytest
import torch

from transformertf.data import EncoderDecoderDataModule, TransformerDataModule
from transformertf.data.datamodule._base import TIME_PREFIX


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
    # Time starts with padding value 0.0, then scaled values in [-1, 1]
    assert sample["encoder_input"][0, 0] == 0.0  # time starts with padding
    # Check that all non-padding time values are in [-1, 1] range
    # When there's no variance in time data, values may all be mapped to center (0.0)
    time_values = sample["encoder_input"][:, 0]
    non_padding_mask = time_values != 0.0  # Assuming padding is 0.0
    if torch.any(non_padding_mask):
        non_padding_times = time_values[non_padding_mask]
        assert torch.all(non_padding_times >= -1.0)
        assert torch.all(non_padding_times <= 1.0)
    else:
        # All time values are 0.0 (either padding or center-mapped due to no variance)
        # This is acceptable behavior for constant time data
        pass

    std = torch.std(sample["encoder_input"][:, 0])
    assert std <= 1.0, f"Standard deviation of time is {std}"

    # ensure that all time values are in [-1, 1] range and std values are reasonable
    for i, sample in enumerate(dataset):
        assert (sample["encoder_input"][:, 0] >= -1.0).all(), (
            f"Time is below -1.0 at index {i}"
        )
        assert (sample["encoder_input"][:, 0] <= 1.0).all(), (
            f"Time is above 1.0 at index {i}"
        )
        std = torch.std(sample["encoder_input"][:, 0])
        assert (std <= 1.2).all(), f"Standard deviation of time is {std}"

        # same for decoder input
        assert (sample["decoder_input"][:, 0] >= -1.0).all(), (
            f"Time is below -1.0 at index {i}"
        )
        assert (sample["decoder_input"][:, 0] <= 1.0).all(), (
            f"Time is above 1.0 at index {i}"
        )
        std = torch.std(sample["decoder_input"][:, 0])
        assert (std <= 1.2).all(), f"Standard deviation of time is {std}"


def test_relative_time_dataset_zero_padded(
    relative_time_datamodule: EncoderDecoderDataModule,
) -> None:
    """Test that last decoder input is zero-padded"""
    relative_time_datamodule.prepare_data()
    relative_time_datamodule.setup()

    dataset = relative_time_datamodule.val_dataset
    sample = dataset[-1]

    assert (sample["decoder_input"][-1, :] == 0.0).all()  # zero-padded decoder input
    assert sample["target"][-1] == 0.0  # zero-padded target

    # ensure that all time values are in [-1, 1] range and std values are reasonable
    for i, sample in enumerate(dataset):
        assert (sample["encoder_input"][:, 0] >= -1.0).all(), (
            f"Time is below -1.0 at index {i}"
        )
        assert (sample["encoder_input"][:, 0] <= 1.0).all(), (
            f"Time is above 1.0 at index {i}"
        )
        std = torch.std(sample["encoder_input"][:, 0])
        assert (std <= 1.2).all(), f"Standard deviation of time is {std}"

        # same for decoder input
        assert (sample["decoder_input"][:, 0] >= -1.0).all(), (
            f"Time is below -1.0 at index {i}"
        )
        assert (sample["decoder_input"][:, 0] <= 1.0).all(), (
            f"Time is above 1.0 at index {i}"
        )
        std = torch.std(sample["decoder_input"][:, 0])
        assert (std <= 1.2).all(), f"Standard deviation of time is {std}"


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
    # Absolute time uses only MinMaxScaler without Delta, so the first time value should be the minimum scaled value
    first_time_val = sample["encoder_input"][0, 0].item()
    assert first_time_val >= -1.0
    assert first_time_val <= 1.0
    assert (
        torch.diff(sample["encoder_input"][:, 0]) >= 0.0
    ).all()  # time is non-decreasing
    assert (
        torch.max(sample["encoder_input"][:, 0]) <= 1.0
    )  # time is normalized to [-1, 1]
    assert (
        torch.min(sample["encoder_input"][:, 0]) >= -1.0
    )  # time is normalized to [-1, 1]


def test_absolute_time_dataset_zero_padded_train(
    absolute_time_datamodule: EncoderDecoderDataModule,
) -> None:
    """Test that last decoder input is zero-padded"""
    absolute_time_datamodule.prepare_data()
    absolute_time_datamodule.setup()

    dataset = absolute_time_datamodule.train_dataset
    sample = dataset[-1]

    assert (
        sample["decoder_input"][-1, :] != 0.0
    ).any()  # not zero-padded decoder input
    assert sample["target"][-1] != 0.0  # not zero-padded target


def test_absolute_time_dataset_zero_padded_val(
    absolute_time_datamodule: EncoderDecoderDataModule,
) -> None:
    """Test that last decoder input is zero-padded"""
    absolute_time_datamodule.prepare_data()
    absolute_time_datamodule.setup()

    dataset = absolute_time_datamodule.val_dataset
    sample = dataset[-1]

    assert (sample["decoder_input"][-1, :] == 0.0).all()  # zero-padded decoder input
    assert sample["target"][-1] == 0.0  # zero-padded target


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
    encoder_lengths = sample["encoder_lengths"].item()
    seq_start = int(datamodule.hparams["ctxt_seq_len"] - encoder_lengths)

    first_time_val = sample["encoder_input"][seq_start, 0].item()
    assert first_time_val >= -1.0
    assert first_time_val <= 1.0
    assert (sample["encoder_input"][:seq_start, 0] == 0.0).all()  # time is zero-padded
    assert (
        sample["encoder_input"][seq_start:, 0] >= -1.0
    ).all()  # time is in [-1, 1] range
    assert (
        torch.max(sample["encoder_input"][:, 0]) <= 1.0
    )  # time is normalized to [-1, 1]
    assert (
        torch.min(sample["encoder_input"][seq_start:, 0]) >= -1.0
    )  # time is normalized to [-1, 1]
