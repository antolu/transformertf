"""
Test the transformertf.data.WindowGenerator class.
"""
from __future__ import annotations

import pytest

from transformertf.data import WindowGenerator


@pytest.mark.parametrize(
    "num_points, window_size, stride, zero_pad, expected",
    [
        (
            10,
            5,
            1,
            False,
            [
                slice(0, 5),
                slice(1, 6),
                slice(2, 7),
                slice(3, 8),
                slice(4, 9),
                slice(5, 10),
            ],
        ),
        (10, 5, 2, False, [slice(0, 5), slice(2, 7), slice(4, 9)]),
        (10, 5, 3, False, [slice(0, 5), slice(3, 8)]),
        (10, 5, 4, False, [slice(0, 5), slice(4, 9)]),
        (10, 5, 5, False, [slice(0, 5), slice(5, 10)]),
        (10, 5, 6, False, [slice(0, 5)]),
        (
            10,
            5,
            1,
            True,
            [
                slice(0, 5),
                slice(1, 6),
                slice(2, 7),
                slice(3, 8),
                slice(4, 9),
                slice(5, 10),
            ],
        ),
        (
            10,
            5,
            2,
            True,
            [slice(0, 5), slice(2, 7), slice(4, 9), slice(6, 11)],
        ),
        (10, 5, 3, True, [slice(0, 5), slice(3, 8), slice(6, 11)]),
        (10, 5, 4, True, [slice(0, 5), slice(4, 9), slice(8, 13)]),
        (10, 5, 5, True, [slice(0, 5), slice(5, 10)]),
        (10, 5, 6, True, [slice(0, 5), slice(6, 11)]),
    ],
)
def test_window_generator_correct_slices(
    num_points: int,
    window_size: int,
    stride: int,
    zero_pad: bool,
    expected: list[slice],
) -> None:
    wg = WindowGenerator(num_points, window_size, stride, zero_pad)
    assert wg[:] == expected


def test_window_generator_access_int() -> None:
    wg = WindowGenerator(10, 5, 1, False)

    assert wg[0] == slice(0, 5)
    assert wg[1] == slice(1, 6)
    assert wg[2] == slice(2, 7)
    assert wg[3] == slice(3, 8)
    assert wg[4] == slice(4, 9)
    assert wg[5] == slice(5, 10)


def test_window_generator_access_slice() -> None:
    wg = WindowGenerator(10, 5, 1, False)

    assert len(wg) == 6

    assert wg[0:1] == [slice(0, 5)]
    assert wg[1:3] == [slice(1, 6), slice(2, 7)]
    assert wg[2:4] == [slice(2, 7), slice(3, 8)]
    assert wg[3:5] == [slice(3, 8), slice(4, 9)]
    assert wg[4:6] == [slice(4, 9), slice(5, 10)]


def test_window_generator_iterable() -> None:
    wg = WindowGenerator(10, 5, 1, False)

    assert list(wg) == [
        slice(0, 5),
        slice(1, 6),
        slice(2, 7),
        slice(3, 8),
        slice(4, 9),
        slice(5, 10),
    ]


def test_window_generator_real_data_len() -> None:
    wg = WindowGenerator(10, 5, 1, False)
    assert wg.real_data_len == 10

    wg = WindowGenerator(10, 5, 1, True)
    assert wg.real_data_len == 11


def test_window_generator_str() -> None:
    wg = WindowGenerator(10, 5, 1, False)
    assert str(wg).startswith("WindowGenerator")
