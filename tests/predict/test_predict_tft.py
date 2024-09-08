from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
from hystcomp_utils.cycle_data import CycleData

from sps_mlp_hystcomp.predict import TFTPredictor


@pytest.fixture(scope="module")
def tft_model_checkpoint_path() -> str:
    sample_data_path = pathlib.Path(__file__).parent / "sample_data"
    checkpoint_file = sample_data_path / "tft_checkpoint.ckpt"

    return os.fspath(checkpoint_file)


@pytest.fixture(scope="module")
def tft_predictor(tft_model_checkpoint_path: str) -> TFTPredictor:
    return TFTPredictor.load_from_checkpoint(tft_model_checkpoint_path, device="cpu")


def test_tft_set_cycled_initial_state(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    tft_predictor.set_cycled_initial_state(buffer)


def test_tft_predict(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    past_covariates = TFTPredictor.buffer_to_covariates(buffer[:-1], rdp=True)

    tft_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = TFTPredictor.buffer_to_covariates([buffer[-1]], rdp=True)
    b_pred_future = tft_predictor.predict(future_covariates)

    t_prog_future = buffer[-1].current_prog[0] / 1e3
    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_tft_predict_cycle(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    # set initial state
    tft_predictor.set_cycled_initial_state(buffer[:-1])

    b_pred_future = tft_predictor.predict_cycle(buffer[-1])

    t_prog_future = buffer[-1].current_prog[0] / 1e3
    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_tft_predict_lasy_cycle(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    b_pred_future = tft_predictor.predict_last_cycle(buffer)

    t_prog_future = buffer[-1].current_prog[0] / 1e3
    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1
