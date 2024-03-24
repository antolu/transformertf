# type: ignore

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.config import TimeSeriesBaseConfig
from transformertf.data import TimeSeriesDataModule
from transformertf.models.phylstm import PhyLSTMConfig, PhyLSTMDataModule
from transformertf.utils import ops

from ...conftest import CURRENT, DF_PATH, FIELD

config = TimeSeriesBaseConfig(input_columns=CURRENT, target_column=FIELD)


@pytest.fixture(scope="module")
def dm() -> TimeSeriesDataModule:
    dm = TimeSeriesDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        known_covariates_cols=[CURRENT],
        target_col=FIELD,
        seq_len=500,
        dtype=torch.float64,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_data_transform_inverse_transform(dm: TimeSeriesDataModule) -> None:
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
    config = PhyLSTMConfig(known_covariates_cols=CURRENT, target_column=FIELD)
    dm = PhyLSTMDataModule.from_parquet(
        config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        downsample=100,
        lowpass_filter=False,
        mean_filter=False,
        dtype=torch.float64,
        target_depends_on=CURRENT,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_phylstm_data_transform_inverse_transform(
    phylstm_dm: PhyLSTMDataModule,
) -> None:
    df = pd.read_parquet(DF_PATH)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[[CURRENT, FIELD]].iloc[:: phylstm_dm.hparams["downsample"]]

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
