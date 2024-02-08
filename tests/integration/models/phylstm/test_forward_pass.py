from __future__ import annotations

import pytest
import torch

from transformertf.models.phylstm import PhyLSTMConfig, PhyLSTMModule


@pytest.fixture(scope="module")
def phylstm_module() -> PhyLSTMModule:
    return PhyLSTMModule.from_config(
        PhyLSTMConfig(num_layers=1, hidden_size=10, hidden_size_fc=16)
    )


def test_phylstm_forward_pass(phylstm_module: PhyLSTMModule) -> None:
    x = torch.rand(1, PhyLSTMConfig.seq_len, 1)

    with torch.no_grad():
        y = phylstm_module(x)

    assert y["z"].shape == (1, PhyLSTMConfig.seq_len, 3)
