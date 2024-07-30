# type: ignore

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformertf.data import TimeSeriesDataModule
from transformertf.utils import ops

from ...conftest import CURRENT, FIELD


@pytest.fixture(scope="module")
def dm(df_path: str, current_key: str, field_key: str) -> TimeSeriesDataModule:
    dm = TimeSeriesDataModule(
        train_df_paths=df_path,
        val_df_paths=df_path,
        known_covariates=[current_key],
        target_covariate=field_key,
        seq_len=500,
        dtype="float64",
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_data_transform_inverse_transform(
    dm: TimeSeriesDataModule, df: pd.DataFrame
) -> None:
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
    y = target_transform.inverse_transform(y)

    x = x.numpy().flatten()
    y = y.numpy().flatten()

    x_true = df[CURRENT].to_numpy()
    y_true = df[FIELD].to_numpy()

    assert np.allclose(x, x_true)
    assert np.allclose(y, y_true)


@pytest.fixture(scope="module")
def phylstm_dm(
    df_path: str,
    current_key: str,
    field_key: str,
) -> TimeSeriesDataModule:
    dm = TimeSeriesDataModule(
        train_df_paths=df_path,
        val_df_paths=df_path,
        downsample=100,
        dtype="float32",
        known_covariates=current_key,
        target_covariate=field_key,
    )
    dm.prepare_data()
    dm.setup()

    return dm


def test_phylstm_data_transform_inverse_transform(
    phylstm_dm: TimeSeriesDataModule,
    df_path: str,
    current_key: str,
    field_key: str,
) -> None:
    df = pd.read_parquet(df_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[[current_key, field_key]].iloc[:: phylstm_dm.hparams["downsample"]]

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

    input_transform = phylstm_dm.input_transforms[current_key]
    target_transform = phylstm_dm.target_transform

    x = input_transform.inverse_transform(x)
    y = target_transform.inverse_transform(y)
    x = x.numpy().flatten()
    y = y.numpy().flatten()

    x_true = df[current_key].to_numpy()
    y_true = df[field_key].to_numpy()

    assert np.allclose(x, x_true)
    assert np.allclose(y, y_true, atol=1e-5)
    # y gets cast to float32 in dataset, so we need to increase atol
