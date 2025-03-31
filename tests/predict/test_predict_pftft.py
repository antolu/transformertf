from __future__ import annotations

import os
import pathlib

import numpy as np
import pybind11_rdp
import pytest
from hystcomp_utils.cycle_data import CycleData

from sps_mlp_hystcomp import PFTFTPredictor


@pytest.fixture(scope="module")
def pftft_model_checkpoint_path() -> str:
    # checkpoint using RDP
    sample_data_path = pathlib.Path(__file__).parent / "sample_data"
    checkpoint_file = sample_data_path / "TFTMBI-77.ckpt"

    return os.fspath(checkpoint_file)


@pytest.fixture(scope="module")
def pftft_predictor(pftft_model_checkpoint_path: str) -> PFTFTPredictor:
    return PFTFTPredictor.load_from_checkpoint(
        pftft_model_checkpoint_path, device="cpu"
    )


def test_pftft_set_cycled_initial_state(
    buffers: list[list[CycleData]], pftft_predictor: PFTFTPredictor
) -> None:
    buffer = buffers[0]

    pftft_predictor.set_cycled_initial_state(buffer)


def test_pftft_predict(
    buffers: list[list[CycleData]], pftft_predictor: PFTFTPredictor
) -> None:
    buffer = buffers[0]

    past_covariates = PFTFTPredictor.buffer_to_covariates(
        buffer[:-1],
        rdp=pftft_predictor.rdp_eps,
        interpolate=True,
        add_past_target=False,
    )

    pftft_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = PFTFTPredictor.buffer_to_covariates(
        [buffer[-1]],
        rdp=pftft_predictor.rdp_eps,
        interpolate=True,
        add_past_target=False,
    )
    b_pred_future = pftft_predictor.predict(future_covariates)

    downsample_factor = (
        pftft_predictor.hparams["downsample"] * pftft_predictor.hparams["stride"]
    )
    t_prog_future = (
        np.arange(len(b_pred_future) * downsample_factor)[::downsample_factor] / 1e3
    )
    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_pftft_predict_rdp(
    buffers: list[list[CycleData]], pftft_predictor: PFTFTPredictor
) -> None:
    buffer = buffers[0]

    pftft_predictor.rdp_eps = 1e-5

    past_covariates = PFTFTPredictor.buffer_to_covariates(
        buffer[:-1],
        rdp=pftft_predictor.rdp_eps,
        interpolate=False,
        add_past_target=True,
    )

    pftft_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = PFTFTPredictor.buffer_to_covariates(
        [buffer[-1]],
        rdp=pftft_predictor.rdp_eps,
        interpolate=False,
    )
    b_pred_future = pftft_predictor.predict(future_covariates, downsample=False)

    i_prog_future = np.vstack(
        (
            buffer[-1].current_prog[0] / 1e3,
            buffer[-1].current_prog[1],
        )
    )
    t_prog_future = pybind11_rdp.rdp(
        i_prog_future.T, epsilon=pftft_predictor.rdp_eps
    ).T[0]

    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_prog_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 1e-3


def test_pftft_predict_cycle(
    buffers: list[list[CycleData]], pftft_predictor: PFTFTPredictor
) -> None:
    buffer = buffers[0]

    # set initial state
    pftft_predictor.set_cycled_initial_state(buffer[:-1])

    t_pred_future, b_pred_future = pftft_predictor.predict_cycle(buffer[-1])

    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_pred_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_pftft_predict_last_cycle(
    buffers: list[list[CycleData]], pftft_predictor: PFTFTPredictor
) -> None:
    buffer = buffers[0]

    t_pred_future, b_pred_future = pftft_predictor.predict_last_cycle(buffer)

    b_meas_future = buffer[-1].field_meas.flatten()
    b_meas_future = np.interp(
        t_pred_future, np.arange(len(b_meas_future)) / 1e3, b_meas_future
    )

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1
