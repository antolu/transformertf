from __future__ import annotations

import pandas as pd
from transformertf.data import TimeSeriesDataset
import math
from ...conftest import CURRENT, FIELD


def test_dataset_train_without_der(df: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=df[CURRENT].values,
        target_data=df[FIELD].values,
        seq_len=500,
        stride=1,
    )

    assert dataset.num_points == len(df)
    assert len(dataset) == len(df) - 500 + 1

    sample = dataset[0]

    assert sample["input"].shape == (500, 1)
    assert sample["target"].shape == (500, 1)

    last_sample = dataset[-1]
    assert last_sample["input"].shape == (500, 1)
    assert last_sample["target"].shape == (500, 1)


    df = df.copy()

    dataset = TimeSeriesDataset(
        input_data=df[CURRENT].values,
        target_data=df[[FIELD]].values,
        seq_len=500,
        stride=1,
    )

    assert dataset.num_points == len(df)
    assert len(dataset) == len(df) - 500 + 1

    sample = dataset[0]

    assert sample["input"].shape == (500, 1)
    assert sample["target"].shape == (500, 1)

    last_sample = dataset[-1]
    assert last_sample["input"].shape == (500, 1)
    assert last_sample["target"].shape == (500, 1)


def test_dataset_val_without_der(df: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=df[CURRENT].values,
        target_data=df[FIELD].values,
        seq_len=500,
        stride=1,
        predict=True,
    )

    assert dataset.num_points == len(df)
    assert len(dataset) == math.ceil(len(df) / 500)

    sample = dataset[0]

    assert sample["input"].shape == (500, 1)
    assert sample["target"].shape == (500, 1)

    last_sample = dataset[-1]
    assert last_sample["input"].shape == (500, 1)
    assert last_sample["target"].shape == (500, 1)
