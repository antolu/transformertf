"""
Tests for the transformertf.data.TimeSeriesSamplesGenerator class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformertf.data import TimeSeriesSampleGenerator


@pytest.mark.parametrize(
    ("arr", "win_size", "stride", "zero_pad", "expected", "sample_shape"),
    [
        (pd.Series(range(9)), 3, 1, False, 7, (3,)),
        (pd.Series(range(9)), 3, 1, True, 7, (3,)),
        (pd.Series(range(7)), 3, 2, False, 3, (3,)),
        (pd.Series(range(7)), 3, 2, True, 3, (3,)),
        (pd.Series(range(8)), 3, 2, False, 3, (3,)),
        (pd.Series(range(8)), 3, 2, True, 4, (3,)),
        (pd.Series(range(16096)), 800, 800, False, 20, (800,)),
        (pd.Series(range(16096)), 800, 800, True, 21, (800,)),
    ],
)
def test_num_samples(
    arr: pd.Series,
    win_size: int,
    stride: int,
    zero_pad: bool,
    expected: int,
    sample_shape: tuple[int, ...],
) -> None:
    wg = TimeSeriesSampleGenerator(arr, win_size, stride=stride, zero_pad=zero_pad)

    assert len(wg) == expected
    last_sample = wg[-1]
    assert isinstance(last_sample, dict)

    for key in ("input",):
        assert key in last_sample

    assert last_sample["input"].shape == sample_shape


# Test 1D data


def test_create_window_generator_xy_1d(x_data: pd.Series, y_data: pd.Series) -> None:
    TimeSeriesSampleGenerator(
        x_data,
        3,
        y_data,
    )


def test_create_window_generator_x_1d(x_data: pd.Series) -> None:
    TimeSeriesSampleGenerator(x_data, 3)


def test_time_series_sample_generator_correct_num_samples_x_1d(
    x_data: pd.Series,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3)
    assert len(wg) == 7


def test_time_series_sample_generator_correct_num_samples_xy_1d(
    x_data: pd.Series, y_data: pd.Series
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data)
    assert len(wg) == 7


def test_time_series_sample_generator_correct_num_samples_x_1d_stride_2(
    x_data: pd.Series,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_xy_1d_stride_2(
    x_data: pd.Series, y_data: pd.Series
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data, stride=2)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_x_1d_stride_2_zero_pad(
    x_data: pd.Series,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2, zero_pad=True)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_xy_1d_stride_2_zero_pad(
    x_data: pd.Series, y_data: pd.Series
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data, stride=2, zero_pad=True)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_samples_x_1d(
    x_data: pd.Series,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3)

    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [2, 3, 4])
    assert np.all(wg[2]["input"] == [3, 4, 5])
    assert np.all(wg[3]["input"] == [4, 5, 6])
    assert np.all(wg[4]["input"] == [5, 6, 7])
    assert np.all(wg[5]["input"] == [6, 7, 8])
    assert np.all(wg[6]["input"] == [7, 8, 9])


def test_time_series_sample_generator_correct_samples_x_1d_stride_2(
    x_data: pd.Series,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2)

    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [3, 4, 5])
    assert np.all(wg[2]["input"] == [5, 6, 7])
    assert np.all(wg[3]["input"] == [7, 8, 9])


def test_time_series_sample_generator_correct_samples_x_1d_stride_2_zero_pad(
    x_data: pd.Series,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2, zero_pad=True)

    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [3, 4, 5])
    assert np.all(wg[2]["input"] == [5, 6, 7])
    assert np.all(wg[3]["input"] == [7, 8, 9])


def test_time_series_sample_generator_correct_samples_xy_1d(
    x_data: pd.Series, y_data: pd.Series
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data)

    # x
    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [2, 3, 4])
    assert np.all(wg[2]["input"] == [3, 4, 5])
    assert np.all(wg[3]["input"] == [4, 5, 6])
    assert np.all(wg[4]["input"] == [5, 6, 7])
    assert np.all(wg[5]["input"] == [6, 7, 8])
    assert np.all(wg[6]["input"] == [7, 8, 9])

    # y
    assert np.all(wg[0]["target"] == [10, 20, 30])
    assert np.all(wg[1]["target"] == [20, 30, 40])
    assert np.all(wg[2]["target"] == [30, 40, 50])
    assert np.all(wg[3]["target"] == [40, 50, 60])
    assert np.all(wg[4]["target"] == [50, 60, 70])
    assert np.all(wg[5]["target"] == [60, 70, 80])
    assert np.all(wg[6]["target"] == [70, 80, 90])


def test_time_series_sample_generator_correct_samples_xy_1d_stride_2(
    x_data: pd.Series, y_data: pd.Series
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data, stride=2)

    # x
    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [3, 4, 5])
    assert np.all(wg[2]["input"] == [5, 6, 7])
    assert np.all(wg[3]["input"] == [7, 8, 9])

    # y
    assert np.all(wg[0]["target"] == [10, 20, 30])
    assert np.all(wg[1]["target"] == [30, 40, 50])
    assert np.all(wg[2]["target"] == [50, 60, 70])
    assert np.all(wg[3]["target"] == [70, 80, 90])


def test_time_series_sample_generator_correct_samples_xy_1d_stride_2_zero_pad(
    x_data: pd.Series, y_data: pd.Series
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data, stride=2, zero_pad=True)

    # x
    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [3, 4, 5])
    assert np.all(wg[2]["input"] == [5, 6, 7])
    assert np.all(wg[3]["input"] == [7, 8, 9])

    # y
    assert np.all(wg[0]["target"] == [10, 20, 30])
    assert np.all(wg[1]["target"] == [30, 40, 50])
    assert np.all(wg[2]["target"] == [50, 60, 70])
    assert np.all(wg[3]["target"] == [70, 80, 90])
