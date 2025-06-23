from __future__ import annotations

import typing

import pytest

from transformertf.models.xtft_conv import xTFTConv


@pytest.fixture(scope="module")
def xtft_conv_module_config() -> dict[str, typing.Any]:
    return {
        "num_past_features": 2,
        "num_future_features": 1,
        "num_lstm_layers": 2,
        "n_dim_model": 32,
        "downsample_factor": 2,
    }


@pytest.fixture(scope="module")
def xtft_conv_module(xtft_conv_module_config: dict[str, typing.Any]) -> xTFTConv:
    return xTFTConv(**xtft_conv_module_config)
