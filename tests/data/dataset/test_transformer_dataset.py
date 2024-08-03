from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from transformertf.data import EncoderDecoderDataset


@pytest.fixture
def x1_1d() -> pd.DataFrame:
    return pd.DataFrame({"data": np.arange(10)})


@pytest.fixture
def x1_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "col2": [10, 11, 12, 13, 14, 15, 16, 17, 18],
    })


@pytest.fixture
def x2_1d() -> pd.DataFrame:
    return pd.DataFrame({"data": np.arange(start=-10, stop=0)})


@pytest.fixture
def x2_2d() -> pd.DataFrame:
    return pd.DataFrame({
        "col1": [-1, -2, -3, -4, -5, -6, -7, -8, -9],
        "col2": [-10, -11, -12, -13, -14, -15, -16, -17, -18],
    })


@pytest.fixture
def y1_1d() -> pd.DataFrame:
    return pd.DataFrame({"data": np.arange(10) / 10})


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
    return pd.DataFrame({"data": np.arange(start=-10, stop=0) / 10})


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
    dataset = EncoderDecoderDataset(
        input_data=x1_1d, target_data=y1_1d, ctx_seq_len=3, tgt_seq_len=2
    )

    assert len(dataset) == 6
    dataset = typing.cast(EncoderDecoderDataset, dataset)
    assert isinstance(dataset[0], dict)
    assert dataset[0]["encoder_input"].shape == (3, 2)
    assert dataset[0]["encoder_mask"].shape == (3, 2)
    assert dataset[0]["decoder_input"].shape == (2, 2)
    assert dataset[0]["decoder_mask"].shape == (2, 2)
    assert dataset[0]["target"].shape == (2, 1)


def test_dataset_train_1d_multiple(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = EncoderDecoderDataset(
        input_data=[x1_1d, x1_1d],
        target_data=[y1_1d, y1_1d],
        ctx_seq_len=3,
        tgt_seq_len=2,
    )

    assert len(dataset) == 12
    assert isinstance(dataset[0], dict)


def test_dataset_known_past_data(x1_1d: pd.DataFrame, y1_1d: pd.DataFrame) -> None:
    dataset = EncoderDecoderDataset(
        input_data=x1_1d,
        target_data=y1_1d,
        known_past_data=x1_1d,
        ctx_seq_len=3,
        tgt_seq_len=2,
    )

    assert len(dataset) == 6
    assert isinstance(dataset[0], dict)
    assert dataset[0]["encoder_input"].shape == (3, 3)
    assert dataset[0]["encoder_mask"].shape == (3, 3)
    assert dataset[0]["decoder_input"].shape == (2, 3)  # type: ignore[typeddict-item]
    assert dataset[0]["decoder_mask"].shape == (2, 3)  # type: ignore[typeddict-item]
    assert dataset[0]["target"].shape == (2, 1)


def test_dataset_known_past_data_too_long(
    x1_1d: pd.DataFrame, y1_1d: pd.DataFrame
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        EncoderDecoderDataset(
            input_data=x1_1d,
            target_data=y1_1d,
            known_past_data=pd.concat([x1_1d, x1_1d], axis=0),
            ctx_seq_len=3,
            tgt_seq_len=2,
        )
