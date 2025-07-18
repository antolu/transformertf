from __future__ import annotations

import typing

import pytest
import torch

from transformertf.models.tsmixer import TSMixer


@pytest.fixture(scope="module")
def tsmixer_module_config() -> dict[str, typing.Any]:
    return {
        "num_features": 2,
        "num_blocks": 2,
        "n_dim_model": 4,  # "hidden_dim": 4,
        "fc_dim": 16,
        "ctxt_seq_len": 10,
        "tgt_seq_len": 5,
    }


@pytest.fixture(scope="module")
def tsmixer_module(tsmixer_module_config: dict[str, typing.Any]) -> TSMixer:
    return TSMixer(**tsmixer_module_config)


def test_tsmixer_forward_pass(
    tsmixer_module: TSMixer, tsmixer_module_config: dict[str, typing.Any]
) -> None:
    past_covariates = torch.rand(1, tsmixer_module_config["ctxt_seq_len"], 2)
    future_covariates = torch.rand(1, tsmixer_module_config["tgt_seq_len"], 1)

    with torch.no_grad():
        y = tsmixer_module.model(past_covariates, future_covariates)

    assert y.shape == (1, tsmixer_module_config["tgt_seq_len"], 1)
