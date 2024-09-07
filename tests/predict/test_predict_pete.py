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


def test_pete_predict(
    buffers: list[list[CycleData]], pete_model_checkpoint: str
) -> None:
    buffer = buffers[0]

    past_covariates = PETEPredictor.buffer_to_covariates(buffer[:-1])

    predictor = PETEPredictor.load_from_checkpoint(pete_model_checkpoint, device="cpu")
    predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    future_covariates = PETEPredictor.buffer_to_covariates([buffer[-1]])
    assert predictor._datamodule is not None  # noqa: SLF001
    b_meas_future = future_covariates["__target__"].to_numpy()[
        :: predictor._datamodule.hparams["downsample"]  # noqa: SLF001
    ]

    b_pred_future = predictor.predict(future_covariates)

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def test_pete_set_cycled_initial_state(
    buffers: list[list[CycleData]], pete_model_checkpoint: str
) -> None:
    buffer = buffers[0]

    predictor = PETEPredictor.load_from_checkpoint(pete_model_checkpoint, device="cpu")
    predictor.set_cycled_initial_state(buffer[:-1])


def test_pete_predict_cycle(
    buffers: list[list[CycleData]], pete_model_checkpoint: str
) -> None:
    buffer = buffers[0]

    predictor = PETEPredictor.load_from_checkpoint(pete_model_checkpoint, device="cpu")

    # set initial state
    past_covariates = PETEPredictor.buffer_to_covariates(buffer[:-1])
    predictor.set_initial_state(
        past_covariates=past_covariates,
    )

    b_meas_future = buffer[-1].field_meas.flatten()

    b_pred_future = predictor.predict_cycle(buffer[-1])
    _t_future, b_pred_future = b_pred_future

    assert predictor._datamodule is not None  # noqa: SLF001
    b_meas_future = b_meas_future[:: predictor._datamodule.hparams["downsample"]]  # noqa: SLF001

    rmse = np.sqrt(np.mean((b_meas_future - b_pred_future) ** 2))
    assert rmse < 0.1


def chain_arrays(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays)
