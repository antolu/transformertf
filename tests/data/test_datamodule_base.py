from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch.utils.data

from transformertf.data import DataModuleBase

DF_PATH = str(Path(__file__).parent.parent / "sample_data.parquet")
CURRENT = "I_meas_A"
FIELD = "B_meas_T"


def test_datamodule_base_create() -> None:
    dm = DataModuleBase(input_columns=["a"], target_columns=["b"])
    assert dm is not None


def test_datamodule_base_create_from_parquet() -> None:
    dm = DataModuleBase(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_columns=[FIELD],
    )
    assert dm is not None


def test_datamodule_base_prepare_data() -> None:
    dm = DataModuleBase(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_columns=[FIELD],
    )
    dm.prepare_data()

    assert dm is not None


def test_datamodule_base_setup_before_prepare_data() -> None:
    dm = DataModuleBase(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_columns=[FIELD],
    )
    dm.setup()

    assert dm is not None

    with pytest.raises(ValueError):
        dm.train_dataset

    with pytest.raises(ValueError):
        dm.val_dataset


@pytest.fixture(scope="module")
def datamodule_base() -> DataModuleBase:
    dm = DataModuleBase(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_columns=[FIELD],
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_datamodule_base_train_dataloader(
    datamodule_base: DataModuleBase,
) -> None:
    dataset = datamodule_base.train_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_base.train_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_base_val_dataloader(
    datamodule_base: DataModuleBase,
) -> None:
    dataset = datamodule_base.val_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = datamodule_base.val_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_datamodule_base_prepare_twice() -> None:
    dm = DataModuleBase(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_columns=[FIELD],
    )
    dm.prepare_data()

    with pytest.raises(RuntimeError):
        dm.prepare_data()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(DF_PATH)


def test_datamodule_base_read_input(
    datamodule_base: DataModuleBase, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.read_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 2


def test_datamodule_base_preprocess_dataframe(
    datamodule_base: DataModuleBase, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.preprocess_dataframe(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == len(df.columns)


def test_datamodule_base_normalize_dataframe(
    datamodule_base: DataModuleBase, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.apply_transforms(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns


def test_datamodule_base_transform_input(
    datamodule_base: DataModuleBase, df: pd.DataFrame
) -> None:
    processed_df = datamodule_base.transform_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 2


def test_datamodule_base_make_dataset(
    datamodule_base: DataModuleBase, df: pd.DataFrame
) -> None:
    dataset = datamodule_base.make_dataset(df)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_datamodule_base_make_dataset_predict(
    datamodule_base: DataModuleBase, df: pd.DataFrame
) -> None:
    dataset = datamodule_base.make_dataset(df, predict=True)

    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)
