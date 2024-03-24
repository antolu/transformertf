from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch.utils.data

from transformertf.config import TimeSeriesBaseConfig
from transformertf.data import TimeSeriesDataModule

DF_PATH = str(Path(__file__).parent.parent / "sample_data.parquet")
CURRENT = "I_meas_A"
FIELD = "B_meas_T"

config = TimeSeriesBaseConfig()


def test_datamodule_timeseries_create_from_parquet() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        known_covariates_cols=["a"],
        target_col="b",
    )
    assert dm is not None


def test_datamodule_timeseries_prepare_data() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        known_covariates_cols=[CURRENT],
        target_col=FIELD,
    )
    dm.prepare_data()

    assert dm is not None


def test_datamodule_timeseries_setup_before_prepare_data() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        known_covariates_cols=[CURRENT],
        target_col=FIELD,
    )
    dm.setup()

    assert dm is not None

    with pytest.raises(ValueError):
        dm.train_dataset

    with pytest.raises(ValueError):
        dm.val_dataset


@pytest.fixture(scope="module")
def datamodule_timeseries() -> TimeSeriesDataModule:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        known_covariates_cols=[CURRENT],
        target_col=FIELD,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_datamodule_timeseries_train_dataloader(
    datamodule_timeseries: TimeSeriesDataModule,
) -> None:
    dataset = datamodule_timeseries.train_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_timeseries.train_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_timeseries_val_dataloader(
    datamodule_timeseries: TimeSeriesDataModule,
) -> None:
    dataset = datamodule_timeseries.val_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_timeseries.val_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_timeseries_prepare_twice() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        known_covariates_cols=[CURRENT],
        target_col=FIELD,
    )
    dm.prepare_data()

    dm.prepare_data()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(DF_PATH)


def test_datamodule_timeseries_read_input(
    datamodule_timeseries: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_timeseries.read_input(
        df, input_columns=[CURRENT], target_column=FIELD
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_timeseries_preprocess_dataframe(
    datamodule_timeseries: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_timeseries.preprocess_dataframe(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == len(df.columns)


def test_datamodule_timeseries_normalize_dataframe(
    datamodule_timeseries: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_timeseries.apply_transforms(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns


def test_datamodule_timeseries_transform_input(
    datamodule_timeseries: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_timeseries.transform_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_timeseries_make_dataset(
    datamodule_timeseries: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    dataset = datamodule_timeseries.make_dataset(df)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_datamodule_timeseries_make_dataset_predict(
    datamodule_timeseries: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    dataset = datamodule_timeseries.make_dataset(df, predict=True)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)
