from __future__ import annotations

import os
import pathlib

import pytest


@pytest.fixture(scope="module")
def mlp_model_checkpoint() -> str:
    sample_data_path = pathlib.Path(__file__).parent.parent / "predict/sample_data"
    checkpoint_file = sample_data_path / "tft_checkpoint_downsample.ckpt"

    return os.fspath(checkpoint_file)
