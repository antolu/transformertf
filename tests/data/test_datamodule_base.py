from __future__ import annotations

import typing

import pandas as pd
import pytest
import torch.utils.data

from transformertf.data import TimeSeriesDataModule


@pytest.fixture(scope="module")
def timeseries_datamodule_config(df_path: str) -> dict[str, typing.Any]:
    return {
        "known_covariates": ["a"],
        "target_covariate": "b",
        "train_df_paths": df_path,
        "val_df_paths": df_path,
        "normalize": True,
        "seq_len": 200,
        "min_seq_len": None,
        "randomize_seq_len": False,
        "stride": 1,
        "downsample": 1,
        "downsample_method": "interval",
        "target_depends_on": None,
        "extra_transforms": None,
        "batch_size": 128,
        "num_workers": 0,
        "dtype": "float32",
        "distributed_sampler": False,
    }


def test_datamodule_base_create_from_df(
    timeseries_datamodule_config: dict[str, typing.Any],
) -> None:
    timeseries_datamodule_config = timeseries_datamodule_config.copy()

    dm = TimeSeriesDataModule(**timeseries_datamodule_config)
    assert dm is not None


def test_datamodule_base_prepare_data(
    timeseries_datamodule_config: dict[str, typing.Any],
    current_key: str,
    field_key: str,
) -> None:
    timeseries_datamodule_config = timeseries_datamodule_config | {
        "known_covariates": [current_key],
        "target_covariate": field_key,
    }
    dm = TimeSeriesDataModule(**timeseries_datamodule_config)
    dm.prepare_data()

    assert dm is not None


def test_datamodule_base_setup_before_prepare_data(
    timeseries_datamodule_config: dict[str, typing.Any],
    current_key: str,
    field_key: str,
) -> None:
    timeseries_datamodule_config = timeseries_datamodule_config | {
        "known_covariates": [current_key],
        "target_covariate": field_key,
    }
    dm = TimeSeriesDataModule(**timeseries_datamodule_config)
    dm.setup()

    assert dm is not None

    with pytest.raises(ValueError):  # noqa: PT011
        _ = dm.train_dataset

    with pytest.raises(ValueError):  # noqa: PT011
        _ = dm.val_dataset


@pytest.fixture(scope="module")
def timeseries_datamodule(
    timeseries_datamodule_config: dict[str, typing.Any],
    current_key: str,
    field_key: str,
) -> TimeSeriesDataModule:
    timeseries_datamodule_config = timeseries_datamodule_config | {
        "known_covariates": [current_key],
        "target_covariate": field_key,
    }
    dm = TimeSeriesDataModule(**timeseries_datamodule_config)
    dm.prepare_data()
    dm.setup()

    return dm


def test_datamodule_base_train_dataloader(
    timeseries_datamodule: TimeSeriesDataModule,
) -> None:
    dataset = timeseries_datamodule.train_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = timeseries_datamodule.train_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_base_val_dataloader(
    timeseries_datamodule: TimeSeriesDataModule,
) -> None:
    dataset = timeseries_datamodule.val_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = timeseries_datamodule.val_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_base_prepare_twice(
    timeseries_datamodule: TimeSeriesDataModule,
) -> None:
    timeseries_datamodule.prepare_data()

    timeseries_datamodule.prepare_data()


def test_datamodule_base_read_input(
    timeseries_datamodule: TimeSeriesDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = timeseries_datamodule.parse_dataframe(
        df, input_columns=[current_key], target_column=field_key
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_base_preprocess_dataframe(
    timeseries_datamodule: TimeSeriesDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = timeseries_datamodule.preprocess_dataframe(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns

    assert len(processed_df.columns) == len(df.columns)


def test_datamodule_base_normalize_dataframe(
    timeseries_datamodule: TimeSeriesDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = timeseries_datamodule.apply_transforms(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns


def test_datamodule_base_transform_input(
    timeseries_datamodule: TimeSeriesDataModule,
    df: pd.DataFrame,
    current_key: str,
    field_key: str,
) -> None:
    processed_df = timeseries_datamodule.transform_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert current_key in processed_df.columns
    assert field_key in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_base_make_dataset(
    timeseries_datamodule: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    dataset = timeseries_datamodule.make_dataset(df)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_datamodule_base_make_dataset_predict(
    timeseries_datamodule: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    dataset = timeseries_datamodule.make_dataset(df, predict=True)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)
