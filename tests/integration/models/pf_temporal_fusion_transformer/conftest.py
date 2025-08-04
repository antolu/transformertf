from __future__ import annotations

import typing

import pytest

from transformertf.models.pf_tft import PFTemporalFusionTransformer


@pytest.fixture(scope="module")
def tft_module_config(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    return {
        "num_past_features": 2,
        "num_future_features": 1,
        "num_lstm_layers": 2,
        "d_model": 32,
    }


@pytest.fixture(scope="module")
def tft_module(tft_module_config: dict[str, typing.Any]) -> PFTemporalFusionTransformer:
    return PFTemporalFusionTransformer(**tft_module_config)
