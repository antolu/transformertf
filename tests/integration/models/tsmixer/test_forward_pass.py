from __future__ import annotations

import pytest
import torch

from transformertf.models.tsmixer import TSMixerConfig, TSMixerModule


@pytest.fixture(scope="module")
def tsmixer_module() -> TSMixerModule:
    return TSMixerModule.from_config(
        TSMixerConfig(), num_features=2, hidden_dim=4
    )


def test_tsmixer_forward_pass(tsmixer_module: TSMixerModule) -> None:
    past_covariates = torch.rand(1, TSMixerConfig.ctxt_seq_len, 2)
    future_covariates = torch.rand(1, TSMixerConfig.tgt_seq_len, 1)

    y = tsmixer_module.model(past_covariates, future_covariates)

    assert y.shape == (1, TSMixerConfig.tgt_seq_len, 1)
