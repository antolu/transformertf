from __future__ import annotations

import typing

import pytest

from transformertf.models.phytsmixer import PhyTSMixer


@pytest.fixture(scope="module")
def phytsmixer_module_config(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    return {
        "num_features": 2,
        "num_blocks": 2,
        "n_dim_model": 4,  # "hidden_dim": 4,
        "fc_dim": 16,
        "ctxt_seq_len": encoder_decoder_datamodule_config["ctxt_seq_len"],
        "tgt_seq_len": encoder_decoder_datamodule_config["tgt_seq_len"],
    }


@pytest.fixture(scope="module")
def phytsmixer_module(phytsmixer_module_config: dict[str, typing.Any]) -> PhyTSMixer:
    return PhyTSMixer(**phytsmixer_module_config)
