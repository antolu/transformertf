from __future__ import annotations

import typing

import pandas as pd
import pytest
import torch.utils.data

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
        "randomize_seq_len": False,
        "stride": 1,
        "downsample": 1,
        "downsample_method": "interval",
        "target_depends_on": None,
        "extra_transforms": None,
        "batch_size": 16,
        "num_workers": 0,
        "dtype": "float32",
        "distributed_sampler": False,
    }


@pytest.fixture(scope="module")
def datamodule_transformer(
    transformer_datamodule_config: dict[str, typing.Any],
) -> EncoderDecoderDataModule:
    dm = EncoderDecoderDataModule(**transformer_datamodule_config)
    dm.prepare_data()
    dm.setup()

    return dm


def test_datamodule_transformer_prepare_data(
    transformer_datamodule_config: dict[str, typing.Any],
) -> None:
    dm = EncoderDecoderDataModule(**transformer_datamodule_config)
    dm.prepare_data()

    assert dm is not None


def test_datamodule_transformer_setup_before_prepare_data(
    transformer_datamodule_config: dict[str, typing.Any],
) -> None:
    dm = EncoderDecoderDataModule(**transformer_datamodule_config)
    dm.setup()

    assert dm is not None

    with pytest.raises(ValueError):  # noqa: PT011
        _ = dm.train_dataset

    with pytest.raises(ValueError):  # noqa: PT011
        _ = dm.val_dataset


def test_datamodule_transformer_train_dataloader(
    datamodule_transformer: EncoderDecoderDataModule,
) -> None:
    dataset = datamodule_transformer.train_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_transformer.train_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_transformer_val_dataloader(
    datamodule_transformer: EncoderDecoderDataModule,
) -> None:
    dataset = datamodule_transformer.val_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_transformer.val_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_transformer_prepare_twice(
    datamodule_transformer: EncoderDecoderDataModule,
) -> None:
    datamodule_transformer.prepare_data()


def test_datamodule_transformer_read_input(
    datamodule_transformer: EncoderDecoderDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = datamodule_transformer.parse_dataframe(
        df, input_columns=[current_key], target_column=field_key
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_transformer_preprocess_dataframe(
    datamodule_transformer: EncoderDecoderDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = datamodule_transformer.preprocess_dataframe(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns

    assert len(processed_df.columns) == len(df.columns)


def test_datamodule_transformer_normalize_dataframe(
    datamodule_transformer: EncoderDecoderDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = datamodule_transformer.apply_transforms(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns


def test_datamodule_transformer_transform_input(
    datamodule_transformer: EncoderDecoderDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = datamodule_transformer.transform_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_transformer_make_dataset(
    datamodule_transformer: EncoderDecoderDataModule, df: pd.DataFrame
) -> None:
    dataset = datamodule_transformer.make_dataset(df)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_datamodule_transformer_make_dataset_predict(
    datamodule_transformer: EncoderDecoderDataModule, df: pd.DataFrame
) -> None:
    dataset = datamodule_transformer.make_dataset(df, predict=True)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)
