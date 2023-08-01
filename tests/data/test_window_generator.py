from __future__ import annotations

import pytest
import numpy as np

from transformerft.data import WindowGenerator


@pytest.fixture(scope="module")
def x_data() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture(scope="module")
def y_data() -> np.ndarray:
    return np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])


@pytest.fixture(scope="module")
def x_data_2d() -> np.ndarray:
    return np.arange(20).reshape((10, 2))


@pytest.fixture(scope="module")
def y_data_2d() -> np.ndarray:
    return np.arange(20, 40).reshape((10, 2))


# Test 1D data


def test_create_window_generator_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    WindowGenerator(
        x_data,
        3,
        y_data,
    )


def test_create_window_generator_x_1d(x_data: np.ndarray) -> None:
    WindowGenerator(x_data, 3)


def test_window_generator_correct_num_samples_x_1d(x_data: np.ndarray) -> None:
    wg = WindowGenerator(x_data, 3)
    assert wg.num_samples == 7


def test_window_generator_correct_num_samples_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data)
    assert wg.num_samples == 7


def test_window_generator_correct_num_samples_x_1d_stride_2(
    x_data: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data, 3, stride=2)
    assert wg.num_samples == 4


def test_window_generator_correct_num_samples_xy_1d_stride_2(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data, stride=2)
    assert wg.num_samples == 4


def test_window_generator_correct_num_samples_x_1d_stride_2_zero_pad(
    x_data: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data, 3, stride=2, zero_pad=True)
    assert wg.num_samples == 4


def test_window_generator_correct_num_samples_xy_1d_stride_2_zero_pad(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data, stride=2, zero_pad=True)
    assert wg.num_samples == 4


def test_window_generator_correct_slices_x_1d(x_data: np.ndarray) -> None:
    wg = WindowGenerator(x_data, 3)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(1, 4)
    assert wg.calc_slice(2) == slice(2, 5)
    assert wg.calc_slice(3) == slice(3, 6)
    assert wg.calc_slice(4) == slice(4, 7)
    assert wg.calc_slice(5) == slice(5, 8)
    assert wg.calc_slice(6) == slice(6, 9)


def test_window_generator_correct_samples_x_1d(x_data: np.ndarray) -> None:
    wg = WindowGenerator(x_data, 3)

    assert np.all(wg[0] == [1, 2, 3])
    assert np.all(wg[1] == [2, 3, 4])
    assert np.all(wg[2] == [3, 4, 5])
    assert np.all(wg[3] == [4, 5, 6])
    assert np.all(wg[4] == [5, 6, 7])
    assert np.all(wg[5] == [6, 7, 8])
    assert np.all(wg[6] == [7, 8, 9])


def test_window_generator_correct_slices_x_1d_stride_2(
    x_data: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data, 3, stride=2)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)


def test_window_generator_correct_samples_x_1d_stride_2(
    x_data: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data, 3, stride=2)

    assert np.all(wg[0] == [1, 2, 3])
    assert np.all(wg[1] == [3, 4, 5])
    assert np.all(wg[2] == [5, 6, 7])
    assert np.all(wg[3] == [7, 8, 9])


def test_window_generator_correct_slices_x_1d_stride_2_zero_pad(
    x_data: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data, 3, stride=2, zero_pad=True)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)


def test_window_generator_correct_samples_x_1d_stride_2_zero_pad(
    x_data: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data, 3, stride=2, zero_pad=True)

    assert np.all(wg[0] == [1, 2, 3])
    assert np.all(wg[1] == [3, 4, 5])
    assert np.all(wg[2] == [5, 6, 7])
    assert np.all(wg[3] == [7, 8, 9])


def test_window_generator_correct_slices_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(1, 4)
    assert wg.calc_slice(2) == slice(2, 5)
    assert wg.calc_slice(3) == slice(3, 6)
    assert wg.calc_slice(4) == slice(4, 7)
    assert wg.calc_slice(5) == slice(5, 8)
    assert wg.calc_slice(6) == slice(6, 9)


def test_window_generator_correct_samples_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data)

    # x
    assert np.all(wg[0][0] == [1, 2, 3])
    assert np.all(wg[1][0] == [2, 3, 4])
    assert np.all(wg[2][0] == [3, 4, 5])
    assert np.all(wg[3][0] == [4, 5, 6])
    assert np.all(wg[4][0] == [5, 6, 7])
    assert np.all(wg[5][0] == [6, 7, 8])
    assert np.all(wg[6][0] == [7, 8, 9])

    # y
    assert np.all(wg[0][1] == [10, 20, 30])
    assert np.all(wg[1][1] == [20, 30, 40])
    assert np.all(wg[2][1] == [30, 40, 50])
    assert np.all(wg[3][1] == [40, 50, 60])
    assert np.all(wg[4][1] == [50, 60, 70])
    assert np.all(wg[5][1] == [60, 70, 80])
    assert np.all(wg[6][1] == [70, 80, 90])


def test_window_generator_correct_slices_xy_1d_stride_2(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data, stride=2)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)


def test_window_generator_correct_samples_xy_1d_stride_2(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data, stride=2)

    # x
    assert np.all(wg[0][0] == [1, 2, 3])
    assert np.all(wg[1][0] == [3, 4, 5])
    assert np.all(wg[2][0] == [5, 6, 7])
    assert np.all(wg[3][0] == [7, 8, 9])

    # y
    assert np.all(wg[0][1] == [10, 20, 30])
    assert np.all(wg[1][1] == [30, 40, 50])
    assert np.all(wg[2][1] == [50, 60, 70])
    assert np.all(wg[3][1] == [70, 80, 90])


def test_window_generator_correct_slices_xy_1d_stride_2_zero_pad(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data, stride=2, zero_pad=True)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)


def test_window_generator_correct_samples_xy_1d_stride_2_zero_pad(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = WindowGenerator(x_data, 3, y_data, stride=2, zero_pad=True)

    # x
    assert np.all(wg[0][0] == [1, 2, 3])
    assert np.all(wg[1][0] == [3, 4, 5])
    assert np.all(wg[2][0] == [5, 6, 7])
    assert np.all(wg[3][0] == [7, 8, 9])

    # y
    assert np.all(wg[0][1] == [10, 20, 30])
    assert np.all(wg[1][1] == [30, 40, 50])
    assert np.all(wg[2][1] == [50, 60, 70])
    assert np.all(wg[3][1] == [70, 80, 90])


# 2D data


def test_create_window_generator_x_2d(x_data_2d: np.ndarray) -> None:
    WindowGenerator(x_data_2d, 3)


def test_create_window_generator_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    WindowGenerator(x_data_2d, 3, y_data_2d)


def test_window_generator_correct_num_samples_x_2d(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3)

    assert len(wg) == 8
    assert wg.num_samples == 8


def test_window_generator_correct_num_samples_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d)

    assert len(wg) == 8
    assert wg.num_samples == 8


def test_window_generator_correct_num_samples_x_2d_stride_2(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3, stride=2)

    assert len(wg) == 4
    assert wg.num_samples == 4


def test_window_generator_correct_num_samples_xy_2d_stride_2(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d, stride=2)

    assert len(wg) == 4
    assert wg.num_samples == 4


def test_window_generator_correct_num_samples_x_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3, stride=2, zero_pad=True)

    assert len(wg) == 5
    assert wg.num_samples == 5


def test_window_generator_correct_num_samples_xy_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d, stride=2, zero_pad=True)

    assert len(wg) == 5
    assert wg.num_samples == 5


def test_window_generator_correct_slices_x_2d(x_data_2d: np.ndarray) -> None:
    wg = WindowGenerator(x_data_2d, 3)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(1, 4)
    assert wg.calc_slice(2) == slice(2, 5)
    assert wg.calc_slice(3) == slice(3, 6)
    assert wg.calc_slice(4) == slice(4, 7)
    assert wg.calc_slice(5) == slice(5, 8)
    assert wg.calc_slice(6) == slice(6, 9)
    assert wg.calc_slice(7) == slice(7, 10)


def test_window_generator_correct_samples_x_2d(x_data_2d: np.ndarray) -> None:
    wg = WindowGenerator(x_data_2d, 3)

    assert np.all(wg[0] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1] == [[2, 3], [4, 5], [6, 7]])
    assert np.all(wg[2] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[3] == [[6, 7], [8, 9], [10, 11]])
    assert np.all(wg[4] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[5] == [[10, 11], [12, 13], [14, 15]])
    assert np.all(wg[6] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[7] == [[14, 15], [16, 17], [18, 19]])


def test_window_generator_correct_slices_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(1, 4)
    assert wg.calc_slice(2) == slice(2, 5)
    assert wg.calc_slice(3) == slice(3, 6)
    assert wg.calc_slice(4) == slice(4, 7)
    assert wg.calc_slice(5) == slice(5, 8)
    assert wg.calc_slice(6) == slice(6, 9)
    assert wg.calc_slice(7) == slice(7, 10)


def test_window_generator_correct_samples_xy_2d(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d)

    assert np.all(wg[0][0] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1][0] == [[2, 3], [4, 5], [6, 7]])
    assert np.all(wg[2][0] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[3][0] == [[6, 7], [8, 9], [10, 11]])
    assert np.all(wg[4][0] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[5][0] == [[10, 11], [12, 13], [14, 15]])
    assert np.all(wg[6][0] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[7][0] == [[14, 15], [16, 17], [18, 19]])

    assert np.all(wg[0][1] == [[20, 21], [22, 23], [24, 25]])
    assert np.all(wg[1][1] == [[22, 23], [24, 25], [26, 27]])
    assert np.all(wg[2][1] == [[24, 25], [26, 27], [28, 29]])
    assert np.all(wg[3][1] == [[26, 27], [28, 29], [30, 31]])
    assert np.all(wg[4][1] == [[28, 29], [30, 31], [32, 33]])
    assert np.all(wg[5][1] == [[30, 31], [32, 33], [34, 35]])
    assert np.all(wg[6][1] == [[32, 33], [34, 35], [36, 37]])
    assert np.all(wg[7][1] == [[34, 35], [36, 37], [38, 39]])


def test_window_generator_correct_slices_x_2d_stride_2(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3, stride=2)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)


def test_window_generator_correct_samples_x_2d_stride_2(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3, stride=2)

    assert np.all(wg[0] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3] == [[12, 13], [14, 15], [16, 17]])


def test_window_generator_correct_slices_xy_2d_stride_2(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d, stride=2)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)


def test_window_generator_correct_samples_xy_2d_stride_2(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d, stride=2)

    assert np.all(wg[0][0] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1][0] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2][0] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3][0] == [[12, 13], [14, 15], [16, 17]])

    assert np.all(wg[0][1] == [[20, 21], [22, 23], [24, 25]])
    assert np.all(wg[1][1] == [[24, 25], [26, 27], [28, 29]])
    assert np.all(wg[2][1] == [[28, 29], [30, 31], [32, 33]])
    assert np.all(wg[3][1] == [[32, 33], [34, 35], [36, 37]])


def test_window_generator_correct_slices_x_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3, stride=2, zero_pad=True)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)
    assert wg.calc_slice(4) == slice(8, 11)


def test_window_generator_correct_samples_x_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray,
) -> None:
    wg = WindowGenerator(x_data_2d, 3, stride=2, zero_pad=True)

    assert np.all(wg[0] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[4] == [[16, 17], [18, 19], [0, 0]])


def test_window_generator_correct_slices_xy_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d, stride=2, zero_pad=True)

    assert wg.calc_slice(0) == slice(0, 3)
    assert wg.calc_slice(1) == slice(2, 5)
    assert wg.calc_slice(2) == slice(4, 7)
    assert wg.calc_slice(3) == slice(6, 9)
    assert wg.calc_slice(4) == slice(8, 11)


def test_window_generator_correct_samples_xy_2d_stride_2_zero_pad(
    x_data_2d: np.ndarray, y_data_2d: np.ndarray
) -> None:
    wg = WindowGenerator(x_data_2d, 3, y_data_2d, stride=2, zero_pad=True)

    assert np.all(wg[0][0] == [[0, 1], [2, 3], [4, 5]])
    assert np.all(wg[1][0] == [[4, 5], [6, 7], [8, 9]])
    assert np.all(wg[2][0] == [[8, 9], [10, 11], [12, 13]])
    assert np.all(wg[3][0] == [[12, 13], [14, 15], [16, 17]])
    assert np.all(wg[4][0] == [[16, 17], [18, 19], [0, 0]])

    assert np.all(wg[0][1] == [[20, 21], [22, 23], [24, 25]])
    assert np.all(wg[1][1] == [[24, 25], [26, 27], [28, 29]])
    assert np.all(wg[2][1] == [[28, 29], [30, 31], [32, 33]])
    assert np.all(wg[3][1] == [[32, 33], [34, 35], [36, 37]])
    assert np.all(wg[4][1] == [[36, 37], [38, 39], [0, 0]])


# Test correct inputs to constructor


def test_window_generator_wrong_array_length_1d() -> None:
    with pytest.raises(ValueError):
        WindowGenerator(np.arange(10), 11)


def test_window_generator_wrong_array_length_2d() -> None:
    with pytest.raises(ValueError):
        WindowGenerator(np.arange(10).reshape(5, 2), 11)


def test_window_generator_wrong_array_length_xy_1d() -> None:
    with pytest.raises(ValueError):
        WindowGenerator(np.arange(10), 3, np.arange(20))


def test_window_generator_wrong_array_length_xy_2d() -> None:
    with pytest.raises(ValueError):
        WindowGenerator(
            np.arange(10).reshape(5, 2), 3, np.arange(22).reshape(11, 2)
        )
