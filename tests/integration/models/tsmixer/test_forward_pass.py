from __future__ import annotations


import pytest
import torch
from transformertf.models.tsmixer import TSMixerModule, TSMixerConfig


@pytest.fixture(scope="module")
def tsmixer_module() -> TSMixerModule:
    return TSMixerModule.from_config(TSMixerConfig())


def test_tsmixer_forward_pass(tsmixer_module: TSMixerModule) -> None:
    x = torch.rand(1, TSMixerConfig.seq_len, 1)

    y = tsmixer_module.model(x)

    assert y.shape == (1, TSMixerConfig.out_seq_len, 1)
