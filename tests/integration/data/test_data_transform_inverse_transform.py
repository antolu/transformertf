# type: ignore
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformertf.data import DataModuleBase
from transformertf.models.phylstm import PhyLSTMDataModule
from transformertf.utils import ops

from ...conftest import CURRENT, DF_PATH, FIELD


@pytest.fixture(scope="module")
def dm() -> DataModuleBase:
    dm = DataModuleBase(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        input_columns=[CURRENT],
        target_columns=[FIELD],
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

    x, y = dataset.inverse_transform(x, y)
    x = x.numpy().flatten()
    y = y.numpy().flatten()

    x_true = df[CURRENT].values
    y_true = df[FIELD].values

    assert np.allclose(x, x_true)
    assert np.allclose(y, y_true)


@pytest.fixture(scope="module")
def physical_dm() -> PhyLSTMDataModule:
    dm = PhyLSTMDataModule(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        polynomial_iterations=10,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_physical_data_transform_inverse_transform(
    physical_dm: PhyLSTMDataModule,
) -> None:
    df = pd.read_parquet(DF_PATH)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[[CURRENT, FIELD]]

    dataset = physical_dm.val_dataset

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

    x, y = dataset.inverse_transform(x, y)
    x = x.numpy().flatten()
    y = y.numpy().flatten()

    x_true = df[CURRENT].values
    y_true = df[FIELD].values

    assert np.allclose(x, x_true)
    assert np.allclose(y, y_true)
