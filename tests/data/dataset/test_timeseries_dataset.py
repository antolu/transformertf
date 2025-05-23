from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformertf.data import TimeSeriesDataset


@pytest.fixture
def x1_1d() -> pd.DataFrame:
    return pd.DataFrame(np.arange(10))


@pytest.fixture
def x1_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "col2": [10, 11, 12, 13, 14, 15, 16, 17, 18],
    })


@pytest.fixture
def x2_1d() -> pd.DataFrame:
    return pd.DataFrame(np.arange(start=-10, stop=0))


@pytest.fixture
def x2_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col1": [-1, -2, -3, -4, -5, -6, -7, -8, -9],
        "col2": [-10, -11, -12, -13, -14, -15, -16, -17, -18],
    })


@pytest.fixture
def y1_1d() -> pd.DataFrame:
    return pd.DataFrame(np.arange(10) / 10)


@pytest.fixture
def y1_2d() -> pd.DataFrame:
    return (
        pd.DataFrame({
            "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "col2": [10, 11, 12, 13, 14, 15, 16, 17, 18],
        })
        / 10
    )


@pytest.fixture
def y2_1d() -> pd.DataFrame:
    return pd.DataFrame(np.arange(start=-10, stop=0) / 10)


@pytest.fixture
def y2_2d() -> pd.DataFrame:
    return (
        pd.DataFrame({
            "col1": [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            "col2": [-10, -11, -12, -13, -14, -15, -16, -17, -18],
        })
        / 10
    )


def test_dataset_train_1d_single(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(input_data=x1_1d, seq_len=3, target_data=y1_1d)

    assert len(dataset) == 8
    assert isinstance(dataset[0], dict)
    assert dataset[0]["input"].shape == (3, 1)
    assert dataset[0]["target"].shape == (3, 1)


def test_dataset_train_1d_multiple(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=[x1_1d, x1_1d], seq_len=3, target_data=[y1_1d, y1_1d]
    )

    assert len(dataset) == 16
    assert isinstance(dataset[0], dict)
    assert dataset[0]["input"].shape == (3, 1)
    assert dataset[0]["target"].shape == (3, 1)


def test_dataset_train_2d_single(x1_2d: pd.DataFrame, y1_2d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(input_data=x1_2d, seq_len=3, target_data=y1_2d)

    assert len(dataset) == 7
    assert isinstance(dataset[0], dict)
    assert dataset[0]["input"].shape == (3, 2)
    assert dataset[0]["target"].shape == (3, 2)


def test_dataset_train_2d_multiple(x1_2d: pd.DataFrame, y1_2d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=[x1_2d, x1_2d], seq_len=3, target_data=[y1_2d, y1_2d]
    )

    assert len(dataset) == 14
    assert isinstance(dataset[0], dict)
    assert dataset[0]["input"].shape == (3, 2)
    assert dataset[0]["target"].shape == (3, 2)


def test_dataset_test_1d_single(x2_1d: pd.DataFrame, y2_1d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=x2_1d, seq_len=3, target_data=y2_1d, predict=True
    )

    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample["input"].shape == (3, 1)
    assert "target" in sample
    assert sample["target"].shape == (3, 1)


def test_dataset_test_1d_multiple(x2_1d: pd.DataFrame, y2_1d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=[x2_1d, x2_1d],
        seq_len=3,
        target_data=[y2_1d, y2_1d],
        predict=True,
    )

    assert len(dataset) == 8
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample["input"].shape == (3, 1)
    assert "target" in sample
    assert sample["target"].shape == (3, 1)


def test_dataset_test_2d_single(x2_2d: pd.DataFrame, y2_2d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=x2_2d, seq_len=3, target_data=y2_2d, predict=True
    )

    assert len(dataset) == 3
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample["input"].shape == (3, 2)
    assert "target" in sample
    assert sample["target"].shape == (3, 2)


def test_dataset_test_2d_multiple(x2_2d: pd.DataFrame, y2_2d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(
        input_data=[x2_2d, x2_2d],
        seq_len=3,
        target_data=[y2_2d, y2_2d],
        predict=True,
    )

    assert len(dataset) == 6
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample["input"].shape == (3, 2)
    assert "target" in sample
    assert sample["target"].shape == (3, 2)


def test_dataset_predict_1d_single(x1_1d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(input_data=x1_1d, seq_len=3, predict=True)

    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample["input"].shape == (3, 1)
    assert "target" not in sample


def test_dataset_predict_1d_multiple(x1_1d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesDataset(input_data=[x1_1d, x1_1d], seq_len=3, predict=True)


def test_dataset_predict_2d_single(x1_2d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(input_data=x1_2d, seq_len=3, predict=True)

    assert len(dataset) == 3
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert sample["input"].shape == (3, 2)
    assert "target" not in sample


def test_dataset_predict_2d_multiple(x1_2d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesDataset(input_data=[x1_2d, x1_2d], seq_len=3, predict=True)


def test_dataset_num_points_1d_single(x1_1d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        dataset = TimeSeriesDataset(input_data=x1_1d, seq_len=3)
    dataset = TimeSeriesDataset(input_data=x1_1d, seq_len=3, predict=True)

    assert dataset.num_points == 10


def test_dataset_num_points_1d_multiple(x1_1d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesDataset(input_data=[x1_1d, x1_1d], seq_len=3)


def test_dataset_num_points_2d_single(x1_2d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        dataset = TimeSeriesDataset(input_data=x1_2d, seq_len=3)

    dataset = TimeSeriesDataset(input_data=x1_2d, seq_len=3, predict=True)
    assert dataset.num_points == 9


def test_dataset_num_points_2d_multiple(x1_2d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesDataset(input_data=[x1_2d, x1_2d], seq_len=3)


def test_dataset_randomize_seq_len_1d_single(
    x1_1d: pd.DataFrame, y1_1d: pd.DataFrame
) -> None:
    dataset = TimeSeriesDataset(
        input_data=x1_1d,
        target_data=y1_1d,
        seq_len=3,
        min_seq_len=1,
        randomize_seq_len=True,
    )

    assert len(dataset) == 8
    assert dataset[0]["input"].shape == (3, 1)
    assert dataset[0]["target"].shape == (3, 1)


def test_dataset_seq_len_none(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = TimeSeriesDataset(input_data=x1_1d, target_data=y1_1d, seq_len=None)

    assert len(dataset) == 1
    assert dataset.num_points == 10


def test_dataset_seq_len_none_2x(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        dataset = TimeSeriesDataset(
            input_data=[x1_1d, x1_1d], target_data=[y1_1d, y1_1d], seq_len=None
        )

    dataset = TimeSeriesDataset(
        input_data=[x1_1d, x1_1d],
        target_data=[y1_1d, y1_1d],
        seq_len=len(x1_1d),
    )
    assert len(dataset) == 2
    assert dataset.num_points == 20
