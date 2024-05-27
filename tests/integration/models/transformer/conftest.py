from __future__ import annotations

import typing

import pytest

from transformertf.models.transformer import VanillaTransformer


@pytest.fixture(scope="module")
def transformer_module_config(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    return {
        "num_features": 2,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "n_dim_model": 32,
        "num_heads": 4,
        "fc_dim": 16,
        "ctxt_seq_len": encoder_decoder_datamodule_config["ctxt_seq_len"],
        "tgt_seq_len": encoder_decoder_datamodule_config["tgt_seq_len"],
    }


@pytest.fixture(scope="module")
def transformer_module(
    transformer_module_config: dict[str, typing.Any],
) -> VanillaTransformer:
    return VanillaTransformer(**transformer_module_config)
