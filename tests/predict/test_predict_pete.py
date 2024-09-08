from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
from hystcomp_utils.cycle_data import CycleData

from sps_mlp_hystcomp.predict import PETEPredictor


@pytest.fixture(scope="module")
def pete_model_checkpoint() -> str:
    sample_data_path = pathlib.Path(__file__).parent / "sample_data"
    checkpoint_file = sample_data_path / "pete_checkpoint.ckpt"

    return os.fspath(checkpoint_file)


@pytest.fixture(scope="module")
def pete_predictor(pete_model_checkpoint: str) -> PETEPredictor:
    return PETEPredictor.load_from_checkpoint(pete_model_checkpoint, device="cpu")


def test_pete_predict(
    buffers: list[list[CycleData]], pete_predictor: PETEPredictor
) -> None:
    buffer = buffers[0]

    past_covariates = PETEPredictor.buffer_to_covariates(buffer[:-1])

    pete_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = PETEPredictor.buffer_to_covariates([buffer[-1]])
    b_meas_future = future_covariates["B_meas_T_filtered"].to_numpy()[
        :: pete_predictor.hparams["downsample"]
    ]

    b_pred_future = pete_predictor.predict(future_covariates)

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_pete_set_cycled_initial_state(
    buffers: list[list[CycleData]], pete_predictor: PETEPredictor
) -> None:
    buffer = buffers[0]

    pete_predictor.set_cycled_initial_state(buffer)


def test_pete_predict_cycle(
    buffers: list[list[CycleData]], pete_predictor: PETEPredictor
) -> None:
    buffer = buffers[0]

    # set initial state
    past_covariates = PETEPredictor.buffer_to_covariates(buffer[:-1])
    pete_predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    b_meas_future = buffer[-1].field_meas.flatten()

    b_pred_future = pete_predictor.predict_cycle(buffer[-1])
    _t_future, b_pred_future = b_pred_future

    b_meas_future = b_meas_future[:: pete_predictor.hparams["downsample"]]

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1
