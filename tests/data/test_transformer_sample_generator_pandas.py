"""
Tests for the transformertf.data.TransformerSampleGenerator class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformertf.data import TransformerSampleGenerator


@pytest.fixture()
def x_data_pd(x_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": x_data})


@pytest.fixture()
def y_data_pd(y_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"y": y_data})


def test_create_window_generator_xy_1d(
    x_data_pd: pd.DataFrame, y_data_pd: pd.DataFrame
) -> None:
    TransformerSampleGenerator(
        x_data_pd,
        y_data_pd,
        3,
        2,
    )


def test_transformer_sample_generator_correct_num_samples_xy_1d(
    x_data_pd: pd.DataFrame, y_data_pd: pd.DataFrame
) -> None:
    wg = TransformerSampleGenerator(x_data_pd, y_data_pd, 3, 2)
    assert len(wg) == 5


def test_transformer_sample_generator_correct_num_samples_xy_1d_stride_2(
    x_data_pd: pd.DataFrame, y_data_pd: pd.DataFrame
) -> None:
    wg = TransformerSampleGenerator(x_data_pd, y_data_pd, 3, 2, stride=2)
    assert len(wg) == 3


def test_transformer_sample_generator_correct_num_samples_xy_1d_stride_2_zero_pad(
    x_data_pd: pd.DataFrame, y_data_pd: pd.DataFrame
) -> None:
    wg = TransformerSampleGenerator(x_data_pd, y_data_pd, 2, 2, stride=2, zero_pad=True)
    assert len(wg) == 4


def test_transformer_sample_generator_keys_xy_1d(
    x_data_pd: pd.DataFrame, y_data_pd: pd.DataFrame
) -> None:
    wg = TransformerSampleGenerator(x_data_pd, y_data_pd, 3, 1)

    assert wg[0].keys() == {
        "encoder_input",
        "encoder_mask",
        "decoder_input",
        "decoder_mask",
        "target",
    }

    assert wg[0]["encoder_input"]["x"].shape == (3,)
    assert len(wg[0]["encoder_input"].columns) == 2
    assert wg[0]["encoder_mask"]["x"].shape == (3,)
    assert len(wg[0]["encoder_mask"].columns) == 2
    assert wg[0]["decoder_input"]["x"].shape == (1,)
    assert len(wg[0]["decoder_input"].columns) == 2
    assert wg[0]["decoder_mask"]["x"].shape == (1,)
    assert len(wg[0]["decoder_mask"].columns) == 2
    assert wg[0]["target"]["y"].shape == (1,)
    assert len(wg[0]["target"].columns) == 1

    assert np.all(wg[0]["encoder_mask"]["x"] == 1)
    assert np.all(wg[0]["decoder_mask"]["x"] == 1)
    assert np.all(wg[0]["decoder_input"][["x", "y"]].to_numpy() == [[4, 0]])
