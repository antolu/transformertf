"""
Tests for downsampling algorithms.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.data import _downsample  # noqa: PLC2701


@pytest.fixture
def arr() -> np.ndarray:
    return np.arange(100)


@pytest.fixture
def arr_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)


@pytest.fixture
def arr_series(arr: np.ndarray) -> pd.Series:
    return pd.Series(arr)


def test_downsample_interval(arr: np.ndarray) -> None:
    assert np.all(_downsample.downsample_array(arr, 2, "interval") == arr[::2])


def test_downsample_interval_torch(arr_torch: torch.Tensor) -> None:
    assert np.equal(
        _downsample.downsample_array(arr_torch, 2, "interval"),
        arr_torch[::2].numpy(),
    ).all()


def test_downsample_interval_series(arr_series: pd.Series) -> None:
    assert np.all(
        _downsample.downsample_array(arr_series, 2, "interval") == arr_series[::2]
    )


def test_downsample_average(arr: np.ndarray) -> None:
    assert _downsample.downsample_array(arr, 2, "median") is not None


def test_downsample_average_torch(arr_torch: torch.Tensor) -> None:
    assert _downsample.downsample_array(arr_torch, 2, "median") is not None


def test_downsample_average_series(arr_series: pd.Series) -> None:
    assert _downsample.downsample_array(arr_series, 2, "median") is not None


def test_downsample_convolve(arr: np.ndarray) -> None:
    assert _downsample.downsample_array(arr, 2, "convolve") is not None


def test_downsample_convolve_torch(arr_torch: torch.Tensor) -> None:
    assert _downsample.downsample_array(arr_torch, 2, "convolve") is not None


def test_downsample_convolve_series(arr_series: pd.Series) -> None:
    assert _downsample.downsample_array(arr_series, 2, "convolve") is not None
