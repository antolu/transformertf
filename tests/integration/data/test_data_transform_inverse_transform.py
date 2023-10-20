# type: ignore

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.config import BaseConfig
from transformertf.data import DataModuleBase
from transformertf.models.phylstm import PhyLSTMConfig, PhyLSTMDataModule
from transformertf.utils import ops

from ...conftest import CURRENT, DF_PATH, FIELD

config = BaseConfig(input_columns=CURRENT, target_column=FIELD)


@pytest.fixture(scope="module")
def dm() -> DataModuleBase:
    dm = DataModuleBase.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_column=FIELD,
        seq_len=500,
        dtype=torch.float64,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_data_transform_inverse_transform(dm: DataModuleBase) -> None:
    df = pd.read_parquet(DF_PATH)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[[CURRENT, FIELD]]

    dataset = dm.val_dataset

    x = [dataset[i]["input"] for i in range(len(dataset))]
    y = [dataset[i]["target"] for i in range(len(dataset))]

    x = ops.concatenate(x)
    y = ops.concatenate(y)

    assert len(x) > len(df)
    assert len(y) > len(df)

    x = ops.truncate(x, dataset.num_points)
    y = ops.truncate(y, dataset.num_points)

    assert len(x) == len(df)
    assert len(y) == len(df)

    input_transform = dm.input_transforms[CURRENT]
    target_transform = dm.target_transform

    x = input_transform.inverse_transform(x)
    y = target_transform.inverse_transform(x, y)

    x = x.numpy().flatten()
    y = y.numpy().flatten()

    x_true = df[CURRENT].values
    y_true = df[FIELD].values

    assert np.allclose(x, x_true)
    assert np.allclose(y, y_true)


@pytest.fixture(scope="module")
def phylstm_dm() -> PhyLSTMDataModule:
    config = PhyLSTMConfig(input_columns=CURRENT, target_column=FIELD)
    dm = PhyLSTMDataModule.from_parquet(
        config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        polynomial_iterations=10,
        dtype=torch.float64,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_physical_data_transform_inverse_transform(
    phylstm_dm: PhyLSTMDataModule,
) -> None:
    df = pd.read_parquet(DF_PATH)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[[CURRENT, FIELD]]

    dataset = phylstm_dm.val_dataset

    x = [dataset[i]["input"] for i in range(len(dataset))]
    y = [dataset[i]["target"] for i in range(len(dataset))]

    x = ops.concatenate(x)
    y = ops.concatenate(y)

    assert len(x) > len(df)
    assert len(y) > len(df)

    x = ops.truncate(x, dataset.num_points)
    y = ops.truncate(y, dataset.num_points)

    assert len(x) == len(df)
    assert len(y) == len(df)

    input_transform = phylstm_dm.input_transforms[CURRENT]
    target_transform = phylstm_dm.target_transform

    x = input_transform.inverse_transform(x)
    y = target_transform.inverse_transform(x, y)
    x = x.numpy().flatten()
    y = y.numpy().flatten()

    x_true = df[CURRENT].values
    y_true = df[FIELD].values

    assert np.allclose(x, x_true)
    assert np.allclose(y, y_true, atol=1e-5)
    # y gets cast to float32 in dataset, so we need to increase atol
