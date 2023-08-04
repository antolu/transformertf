from __future__ import annotations

import pytest
import pandas as pd
from transformertf.data import TimeSeriesDataset
from pathlib import Path
import math


DF_PATH = str(Path(__file__).parent.parent.parent / "sample_data.parquet")
CURRENT = "I_meas_A"
FIELD = "B_meas_T"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    df = pd.read_parquet(DF_PATH)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[[CURRENT, FIELD]]

    return df


def test_dataset_train(df: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=df[CURRENT].values,
        target_data=df[FIELD].values,
        seq_len=500,
        stride=1,
    )

    assert dataset.num_points == len(df)
    assert len(dataset) == len(df) - 500 + 1


def test_dataset_val(df: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=df[CURRENT].values,
        target_data=df[FIELD].values,
        seq_len=500,
        stride=1,
        predict=True,
    )

    assert dataset.num_points == len(df)
    assert len(dataset) == math.ceil(len(df) / 500)
