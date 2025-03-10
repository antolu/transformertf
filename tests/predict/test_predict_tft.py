from __future__ import annotations

import os
import pathlib

import numpy as np
import pybind11_rdp
import pytest
from hystcomp_utils.cycle_data import CycleData

from sps_mlp_hystcomp import TFTPredictor


@pytest.fixture(scope="module")
def tft_model_checkpoint_path() -> str:
    # checkpoint using RDP
    sample_data_path = pathlib.Path(__file__).parent / "sample_data"
    checkpoint_file = sample_data_path / "tft_checkpoint.ckpt"

    return os.fspath(checkpoint_file)


@pytest.fixture(scope="module")
def tft_predictor(tft_model_checkpoint_path: str) -> TFTPredictor:
    return TFTPredictor.load_from_checkpoint(tft_model_checkpoint_path, device="cpu")


@pytest.fixture(scope="module")
def tft_predictor_downsample() -> TFTPredictor:
    sample_data_path = pathlib.Path(__file__).parent / "sample_data"
    checkpoint_file = sample_data_path / "tft_checkpoint_downsample.ckpt"

    return TFTPredictor.load_from_checkpoint(
        os.fspath(checkpoint_file),
        device="cpu",
    )


def test_tft_set_cycled_initial_state(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    tft_predictor.set_cycled_initial_state(buffer)


def test_tft_predict(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    past_covariates = TFTPredictor.buffer_to_covariates(
        buffer[:-1],
        rdp=tft_predictor.rdp_eps,
        interpolate=False,
        add_past_target=True,
    )

    tft_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = TFTPredictor.buffer_to_covariates(
        [buffer[-1]],
        rdp=tft_predictor.rdp_eps,
        interpolate=False,
    )
    b_pred_future = tft_predictor.predict(future_covariates)

    t_prog_future = buffer[-1].current_prog[0] / 1e3
    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_tft_predict_rdp(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    tft_predictor.rdp_eps = 1e-5

    past_covariates = TFTPredictor.buffer_to_covariates(
        buffer[:-1],
        rdp=tft_predictor.rdp_eps,
        interpolate=False,
        add_past_target=True,
    )

    tft_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = TFTPredictor.buffer_to_covariates(
        [buffer[-1]],
        rdp=tft_predictor.rdp_eps,
        interpolate=False,
    )
    b_pred_future = tft_predictor.predict(future_covariates)

    i_prog_future = np.vstack(
        (
            buffer[-1].current_prog[0] / 1e3,
            buffer[-1].current_prog[1],
        )
    )
    t_prog_future = pybind11_rdp.rdp(i_prog_future.T, epsilon=tft_predictor.rdp_eps).T[
        0
    ]

    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 1e-3


def test_tft_predict_downsampled_measured(
    buffers: list[list[CycleData]], tft_predictor_downsample: TFTPredictor
) -> None:
    buffer = buffers[1]

    # tft_predictor.rdp_eps = 1e-5

    t_pred_future, b_pred_future = tft_predictor_downsample.predict_last_cycle(
        buffer,
        use_programmed_current=True,
    )

    np.vstack((buffer[-1].current_prog[0] / 1e3, buffer[-1].current_prog[1]))

    b_meas_future = buffer[-1].field_meas.flatten()
    t_prog_future = np.arange(len(b_meas_future)) / 1e3

    b_pred_future = np.interp(t_prog_future, t_pred_future, b_pred_future)

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 1e-3


def test_tft_predict_cycle(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    # set initial state
    tft_predictor.set_cycled_initial_state(buffer[:-1])

    t_pred_future, b_pred_future = tft_predictor.predict_cycle(buffer[-1])

    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_pred_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_tft_predict_last_cycle(
    buffers: list[list[CycleData]], tft_predictor: TFTPredictor
) -> None:
    buffer = buffers[0]

    t_pred_future, b_pred_future = tft_predictor.predict_last_cycle(buffer)

    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_pred_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1
