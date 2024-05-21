from __future__ import annotations

import pytest
import torch

from transformertf.models.tsmixer import TSMixer, TSMixerConfig


@pytest.fixture(scope="module")
def tsmixer_module() -> TSMixer:
    return TSMixer.from_config(TSMixerConfig(), num_features=2, hidden_dim=4)


def test_tsmixer_forward_pass(tsmixer_module: TSMixer) -> None:
    past_covariates = torch.rand(1, TSMixerConfig.ctxt_seq_len, 2)
    future_covariates = torch.rand(1, TSMixerConfig.tgt_seq_len, 1)

    with torch.no_grad():
        y = tsmixer_module.model(past_covariates, future_covariates)

    assert y.shape == (1, TSMixerConfig.tgt_seq_len, 1)
