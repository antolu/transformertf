import numpy as np
import pandas as pd
import pytest

from transformertf.data import EncoderDataset


@pytest.fixture
def x1_1d() -> pd.DataFrame:
    return pd.DataFrame({"col": np.arange(10)})


@pytest.fixture
def x1_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col0": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "col1": [10, 11, 12, 13, 14, 15, 16, 17, 18],
    })


@pytest.fixture
def x2_1d() -> pd.DataFrame:
    return pd.DataFrame({"col0": np.arange(start=-10, stop=0)})


@pytest.fixture
def x2_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col0": [-1, -2, -3, -4, -5, -6, -7, -8, -9],
        "col1": [-10, -11, -12, -13, -14, -15, -16, -17, -18],
    })


@pytest.fixture
def y1_1d() -> pd.DataFrame:
    return pd.DataFrame({"col0": np.arange(10) / 10})


@pytest.fixture
def y1_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col0": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) / 10,
        "col1": np.array([10, 11, 12, 13, 14, 15, 16, 17, 18]) / 10,
    })


@pytest.fixture
def y2_1d() -> pd.DataFrame:
    return pd.DataFrame({"col0": np.arange(start=-10, stop=0) / 10})


@pytest.fixture
def y2_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col0": np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9]) / 10,
        "col1": np.array([-10, -11, -12, -13, -14, -15, -16, -17, -18]) / 10,
    })


def test_dataset_train_1d_single(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = EncoderDataset(
        input_data=x1_1d, target_data=y1_1d, ctx_seq_len=3, tgt_seq_len=2
    )

    assert len(dataset) == 6
    assert isinstance(dataset[0], dict)
    assert dataset[0]["encoder_input"].shape == (5, 2)
    assert dataset[0]["encoder_mask"].shape == (5, 2)
    assert dataset[0]["target"].shape == (5, 1)


def test_dataset_train_1d_multiple(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = EncoderDataset(
        input_data=[x1_1d, x1_1d],
        target_data=[y1_1d, y1_1d],
        ctx_seq_len=3,
        tgt_seq_len=2,
    )

    assert len(dataset) == 12
    assert isinstance(dataset[0], dict)
