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


def test_datamodule_base_create_from_parquet() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=["a"],
        target_column="b",
    )
    assert dm is not None


def test_datamodule_base_prepare_data() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
    )
    dm.prepare_data()

    assert dm is not None


def test_datamodule_base_setup_before_prepare_data() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
    )
    dm.setup()

    assert dm is not None

    with pytest.raises(ValueError):  # noqa: PT011
        _ = dm.train_dataset

    with pytest.raises(ValueError):  # noqa: PT011
        _ = dm.val_dataset


@pytest.fixture(scope="module")
def datamodule_base() -> TimeSeriesDataModule:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_datamodule_base_train_dataloader(
    datamodule_base: TimeSeriesDataModule,
) -> None:
    dataset = datamodule_base.train_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_base.train_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_base_val_dataloader(
    datamodule_base: TimeSeriesDataModule,
) -> None:
    dataset = datamodule_base.val_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_base.val_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_base_prepare_twice() -> None:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
    )
    dm.prepare_data()

    dm.prepare_data()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(DF_PATH)


def test_datamodule_base_read_input(
    datamodule_base: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.read_input(
        df, input_columns=[CURRENT], target_column=FIELD
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_base_preprocess_dataframe(
    datamodule_base: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.preprocess_dataframe(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == len(df.columns)


def test_datamodule_base_normalize_dataframe(
    datamodule_base: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.apply_transforms(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns


def test_datamodule_base_transform_input(
    datamodule_base: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.transform_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_base_make_dataset(
    datamodule_base: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    dataset = datamodule_base.make_dataset(df)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_datamodule_base_make_dataset_predict(
    datamodule_base: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
    dataset = datamodule_base.make_dataset(df, predict=True)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)
