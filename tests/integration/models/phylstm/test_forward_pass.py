from __future__ import annotations


import pytest
import torch
from transformertf.models.phylstm import PhyLSTMModule, PhyLSTMConfig


@pytest.fixture(scope="module")
def phylstm_module() -> PhyLSTMModule:
    return PhyLSTMModule.from_config(PhyLSTMConfig())


def test_phylstm_forward_pass(phylstm_module: PhyLSTMModule) -> None:
    x = torch.rand(1, PhyLSTMConfig.seq_len, 1)

    y = phylstm_module(x)

    assert y["z"].shape == (1, PhyLSTMConfig.seq_len, 3)
