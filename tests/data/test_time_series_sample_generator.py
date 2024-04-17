"""
Tests for the transformertf.data.TimeSeriesSamplesGenerator class.
"""

from __future__ import annotations

import numpy as np
import pytest

from transformertf.data import TimeSeriesSampleGenerator


@pytest.mark.parametrize(
    ("arr", "win_size", "stride", "zero_pad", "expected", "sample_shape"),
    [
        (np.arange(9), 3, 1, False, 7, (3,)),
        (np.arange(9), 3, 1, True, 7, (3,)),
        (np.arange(7), 3, 2, False, 3, (3,)),
        (np.arange(7), 3, 2, True, 3, (3,)),
        (np.arange(8), 3, 2, False, 3, (3,)),
        (np.arange(8), 3, 2, True, 4, (3,)),
        (np.arange(16096), 800, 800, False, 20, (800,)),
        (np.arange(16096), 800, 800, True, 21, (800,)),
    ],
)
def test_num_samples(
    arr: np.ndarray,
    win_size: int,
    stride: int,
    zero_pad: bool,  # noqa: FBT001
    expected: int,
    sample_shape: tuple[int, ...],
) -> None:
    wg = TimeSeriesSampleGenerator(arr, win_size, stride=stride, zero_pad=zero_pad)

    assert len(wg) == expected
    last_sample = wg[-1]
    assert isinstance(last_sample, dict)

    for key in ("input", "initial_state"):
        assert key in last_sample

    assert last_sample["input"].shape == sample_shape
    assert last_sample["initial_state"].shape == (2,)


# Test 1D data


def test_create_window_generator_xy_1d(x_data: np.ndarray, y_data: np.ndarray) -> None:
    TimeSeriesSampleGenerator(
        x_data,
        3,
        y_data,
    )


def test_create_window_generator_x_1d(x_data: np.ndarray) -> None:
    TimeSeriesSampleGenerator(x_data, 3)


def test_time_series_sample_generator_correct_num_samples_x_1d(
    x_data: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3)
    assert len(wg) == 7


def test_time_series_sample_generator_correct_num_samples_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data)
    assert len(wg) == 7


def test_time_series_sample_generator_correct_num_samples_x_1d_stride_2(
    x_data: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_xy_1d_stride_2(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data, stride=2)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_x_1d_stride_2_zero_pad(
    x_data: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2, zero_pad=True)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_xy_1d_stride_2_zero_pad(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, y_data, stride=2, zero_pad=True)
    assert len(wg) == 4


def test_time_series_sample_generator_correct_samples_x_1d(
    x_data: np.ndarray,
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
    x_data: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2)

    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [3, 4, 5])
    assert np.all(wg[2]["input"] == [5, 6, 7])
    assert np.all(wg[3]["input"] == [7, 8, 9])


def test_time_series_sample_generator_correct_samples_x_1d_stride_2_zero_pad(
    x_data: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data, 3, stride=2, zero_pad=True)

    assert np.all(wg[0]["input"] == [1, 2, 3])
    assert np.all(wg[1]["input"] == [3, 4, 5])
    assert np.all(wg[2]["input"] == [5, 6, 7])
    assert np.all(wg[3]["input"] == [7, 8, 9])


def test_time_series_sample_generator_correct_samples_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
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
    x_data: np.ndarray, y_data: np.ndarray
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
    x_data: np.ndarray, y_data: np.ndarray
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


# 2D data


def test_create_window_generator_x_2d(x_data_2d: np.ndarray) -> None:
    TimeSeriesSampleGenerator(x_data_2d, 3)


def test_create_window_generator_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d)


def test_time_series_sample_generator_correct_num_samples_x_2d(
    x_data_2d: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3)

    assert len(wg) == 8


def test_time_series_sample_generator_correct_num_samples_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d)

    assert len(wg) == 8


def test_time_series_sample_generator_correct_num_samples_x_2d_stride_2(
    x_data_2d: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, stride=2)

    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_xy_2d_stride_2(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d, stride=2)

    assert len(wg) == 4


def test_time_series_sample_generator_correct_num_samples_x_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, stride=2, zero_pad=True)

    assert len(wg) == 5


def test_time_series_sample_generator_correct_num_samples_xy_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d, stride=2, zero_pad=True)

    assert len(wg) == 5


def test_time_series_sample_generator_correct_samples_x_2d(
    x_data_2d: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3)

    assert np.all(wg[0]["input"] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1]["input"] == [[2, 3], [4, 5], [6, 7]])
    assert np.all(wg[2]["input"] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[3]["input"] == [[6, 7], [8, 9], [10, 11]])
    assert np.all(wg[4]["input"] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[5]["input"] == [[10, 11], [12, 13], [14, 15]])
    assert np.all(wg[6]["input"] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[7]["input"] == [[14, 15], [16, 17], [18, 19]])


def test_time_series_sample_generator_correct_samples_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d)

    assert np.all(wg[0]["input"] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1]["input"] == [[2, 3], [4, 5], [6, 7]])
    assert np.all(wg[2]["input"] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[3]["input"] == [[6, 7], [8, 9], [10, 11]])
    assert np.all(wg[4]["input"] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[5]["input"] == [[10, 11], [12, 13], [14, 15]])
    assert np.all(wg[6]["input"] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[7]["input"] == [[14, 15], [16, 17], [18, 19]])

    assert np.all(wg[0]["target"] == [[20, 21], [22, 23], [24, 25]])
    assert np.all(wg[1]["target"] == [[22, 23], [24, 25], [26, 27]])
    assert np.all(wg[2]["target"] == [[24, 25], [26, 27], [28, 29]])
    assert np.all(wg[3]["target"] == [[26, 27], [28, 29], [30, 31]])
    assert np.all(wg[4]["target"] == [[28, 29], [30, 31], [32, 33]])
    assert np.all(wg[5]["target"] == [[30, 31], [32, 33], [34, 35]])
    assert np.all(wg[6]["target"] == [[32, 33], [34, 35], [36, 37]])
    assert np.all(wg[7]["target"] == [[34, 35], [36, 37], [38, 39]])


def test_time_series_sample_generator_correct_samples_x_2d_stride_2(
    x_data_2d: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, stride=2)

    assert np.all(wg[0]["input"] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1]["input"] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2]["input"] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3]["input"] == [[12, 13], [14, 15], [16, 17]])


def test_time_series_sample_generator_correct_samples_xy_2d_stride_2(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d, stride=2)

    assert np.all(wg[0]["input"] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1]["input"] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2]["input"] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3]["input"] == [[12, 13], [14, 15], [16, 17]])

    assert np.all(wg[0]["target"] == [[20, 21], [22, 23], [24, 25]])
    assert np.all(wg[1]["target"] == [[24, 25], [26, 27], [28, 29]])
    assert np.all(wg[2]["target"] == [[28, 29], [30, 31], [32, 33]])
    assert np.all(wg[3]["target"] == [[32, 33], [34, 35], [36, 37]])


def test_time_series_sample_generator_correct_samples_x_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray,
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, stride=2, zero_pad=True)

    assert np.all(wg[0]["input"] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1]["input"] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2]["input"] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3]["input"] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[4]["input"] == [[16, 17], [18, 19], [0, 0]])


def test_time_series_sample_generator_correct_samples_xy_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = TimeSeriesSampleGenerator(x_data_2d, 3, y_data_2d, stride=2, zero_pad=True)

    assert np.all(wg[0]["input"] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1]["input"] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2]["input"] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3]["input"] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[4]["input"] == [[16, 17], [18, 19], [0, 0]])

    assert np.all(wg[0]["target"] == [[20, 21], [22, 23], [24, 25]])
    assert np.all(wg[1]["target"] == [[24, 25], [26, 27], [28, 29]])
    assert np.all(wg[2]["target"] == [[28, 29], [30, 31], [32, 33]])
    assert np.all(wg[3]["target"] == [[32, 33], [34, 35], [36, 37]])
    assert np.all(wg[4]["target"] == [[36, 37], [38, 39], [0, 0]])


# Test correct inputs to constructor


def test_time_series_sample_generator_wrong_array_length_1d() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesSampleGenerator(np.arange(10), 11)


def test_time_series_sample_generator_wrong_array_length_2d() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesSampleGenerator(np.arange(10).reshape(5, 2), 11)


def test_time_series_sample_generator_wrong_array_length_xy_1d() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesSampleGenerator(np.arange(10), 3, np.arange(20))


def test_time_series_sample_generator_wrong_array_length_xy_2d() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TimeSeriesSampleGenerator(
            np.arange(10).reshape(5, 2), 3, np.arange(22).reshape(11, 2)
        )
