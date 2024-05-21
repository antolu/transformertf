from __future__ import annotations

import typing

import pytest

from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer


@pytest.fixture(scope="module")
def tft_module_config(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    return {
        "num_past_features": 2,
        "num_future_features": 1,
        "num_lstm_layers": 2,
        "n_dim_model": 32,
        "ctxt_seq_len": encoder_decoder_datamodule_config["ctxt_seq_len"],
        "tgt_seq_len": encoder_decoder_datamodule_config["tgt_seq_len"],
    }


@pytest.fixture(scope="module")
def tft_module(tft_module_config: dict[str, typing.Any]) -> TemporalFusionTransformer:
    return TemporalFusionTransformer(**tft_module_config)
