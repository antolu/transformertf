from __future__ import annotations

import pytest
import torch

from transformertf.models.phytsmixer import PhyTSMixer


@pytest.fixture
def phytsmixer_module_config() -> dict[str, int]:
    return {
        "num_features": 2,
        "num_blocks": 2,
        "n_dim_model": 4,
        "fc_dim": 16,
        "ctxt_seq_len": 100,
        "tgt_seq_len": 50,
    }


@pytest.fixture
def phytsmixer_module(phytsmixer_module_config: dict[str, int]) -> PhyTSMixer:
    return PhyTSMixer(**phytsmixer_module_config)


def test_phytsmixer_forward_pass(
    phytsmixer_module: PhyTSMixer, phytsmixer_module_config: dict[str, int]
) -> None:
    x_past = torch.rand(1, phytsmixer_module_config["ctxt_seq_len"], 2)
    x_future = torch.rand(1, phytsmixer_module_config["tgt_seq_len"], 1)

    batch = {
        "encoder_input": x_past,
        "decoder_input": torch.cat([x_future, x_future], dim=-1),
    }

    with torch.no_grad():
        y = phytsmixer_module(batch)

    assert y["z"].shape == (1, phytsmixer_module_config["tgt_seq_len"], 3)
