from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch.utils.data

from transformertf.config import TransformerBaseConfig
from transformertf.data import EncoderDecoderDataModule

DF_PATH = str(Path(__file__).parent.parent / "sample_data.parquet")
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
        known_past_columns=[CURRENT],
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_datamodule_transformer_create_from_parquet() -> None:
    dm = EncoderDecoderDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=["a"],
        target_column="b",
    )
    assert dm is not None


def test_datamodule_transformer_prepare_data() -> None:
    dm = EncoderDecoderDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
    )
    dm.prepare_data()

    assert dm is not None


def test_datamodule_transformer_setup_before_prepare_data() -> None:
    dm = EncoderDecoderDataModule.from_parquet(
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


def test_datamodule_transformer_prepare_twice() -> None:
    dm = EncoderDecoderDataModule.from_parquet(
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


def test_datamodule_transformer_read_input(
    datamodule_transformer: EncoderDecoderDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_transformer.read_input(
        df, input_columns=[CURRENT], target_column=FIELD
    )

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == 3


def test_datamodule_transformer_preprocess_dataframe(
    datamodule_transformer: EncoderDecoderDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_transformer.preprocess_dataframe(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

    assert len(processed_df.columns) == len(df.columns)


def test_datamodule_transformer_normalize_dataframe(
    datamodule_transformer: EncoderDecoderDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_transformer.apply_transforms(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns


def test_datamodule_transformer_transform_input(
    datamodule_transformer: EncoderDecoderDataModule, df: pd.DataFrame
) -> None:
    processed_df = datamodule_transformer.transform_input(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)

    assert CURRENT in processed_df.columns
    assert FIELD in processed_df.columns

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
