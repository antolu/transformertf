"""
Tests for the transformertf.data.TransformerSampleGenerator class.
"""
from __future__ import annotations

import numpy as np
import pytest

from transformertf.data import TransformerSampleGenerator


@pytest.fixture(scope="module")
def x_data() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture(scope="module")
def y_data() -> np.ndarray:
    return np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])


def test_create_window_generator_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    TransformerSampleGenerator(
        x_data,
        y_data,
        3,
        2,
    )


def test_transformer_sample_generator_correct_num_samples_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(x_data, y_data, 3, 2)
    assert len(wg) == 5


def test_transformer_sample_generator_correct_num_samples_xy_1d_stride_2(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(x_data, y_data, 3, 2, stride=2)
    assert len(wg) == 3


def test_transformer_sample_generator_correct_num_samples_xy_1d_stride_2_zero_pad(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(
        x_data, y_data, 2, 2, stride=2, zero_pad=True
    )
    assert len(wg) == 4


def test_transformer_sample_generator_correct_samples_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(x_data, y_data, 3, 1)

    # x
    assert np.all(wg[0]["encoder_input"].T == [[1, 2, 3], [10, 20, 30]])
    assert np.all(wg[1]["encoder_input"].T == [[2, 3, 4], [20, 30, 40]])
    assert np.all(wg[2]["encoder_input"].T == [[3, 4, 5], [30, 40, 50]])
    assert np.all(wg[3]["encoder_input"].T == [[4, 5, 6], [40, 50, 60]])
    assert np.all(wg[4]["encoder_input"].T == [[5, 6, 7], [50, 60, 70]])
    assert np.all(wg[5]["encoder_input"].T == [[6, 7, 8], [60, 70, 80]])

    # y
    assert np.all(wg[0]["target"] == [40])
    assert np.all(wg[1]["target"] == [50])
    assert np.all(wg[2]["target"] == [60])
    assert np.all(wg[3]["target"] == [70])
    assert np.all(wg[4]["target"] == [80])
    assert np.all(wg[5]["target"] == [90])


def test_transformer_sample_generator_correct_samples_xy_1d_stride_2(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(x_data, y_data, 3, 1, stride=2)

    # x
    assert np.all(wg[0]["encoder_input"].T == [[1, 2, 3], [10, 20, 30]])
    assert np.all(wg[1]["encoder_input"].T == [[3, 4, 5], [30, 40, 50]])
    assert np.all(wg[2]["encoder_input"].T == [[5, 6, 7], [50, 60, 70]])

    # y
    assert np.all(wg[0]["target"] == [40])
    assert np.all(wg[1]["target"] == [60])
    assert np.all(wg[2]["target"] == [80])

    assert len(wg) == 3


def test_transformer_sample_generator_correct_samples_xy_1d_stride_2_zero_pad(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(
        x_data, y_data, 2, 2, stride=2, zero_pad=True
    )

    # x
    assert np.all(wg[0]["encoder_input"].T == [[1, 2], [10, 20]])
    assert np.all(wg[1]["encoder_input"].T == [[3, 4], [30, 40]])
    assert np.all(wg[2]["encoder_input"].T == [[5, 6], [50, 60]])
    assert np.all(wg[3]["encoder_input"].T == [[7, 8], [70, 80]])

    # y
    assert np.all(wg[0]["target"].T == [30, 40])
    assert np.all(wg[1]["target"].T == [50, 60])
    assert np.all(wg[2]["target"].T == [70, 80])
    assert np.all(wg[3]["target"].T == [90, 0.0])


def test_transformer_sample_generator_keys_xy_1d(
    x_data: np.ndarray, y_data: np.ndarray
) -> None:
    wg = TransformerSampleGenerator(x_data, y_data, 3, 1)

    assert wg[0].keys() == {
        "encoder_input",
        "encoder_mask",
        "decoder_input",
        "decoder_mask",
        "target",
    }

    assert wg[0]["encoder_input"].shape == (3, 2)
    assert wg[0]["encoder_mask"].shape == (3, 2)
    assert wg[0]["decoder_input"].shape == (1, 2)
    assert wg[0]["decoder_mask"].shape == (1, 2)
    assert wg[0]["target"].shape == (1, 1)

    assert np.all(wg[0]["encoder_mask"] == 1)
    assert np.all(wg[0]["decoder_mask"] == 1)
    assert np.all(wg[0]["decoder_input"] == [[4, 0]])
