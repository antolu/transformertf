from __future__ import annotations

import pytest
import torch

from transformertf.models.phytsmixer import PhyTSMixerConfig, PhyTSMixerModule


@pytest.fixture(scope="module")
def phytsmixer_module() -> PhyTSMixerModule:
    return PhyTSMixerModule.from_config(
        PhyTSMixerConfig(input_columns=["I_meas_A"], target_column="B_meas_T")
    )


def test_phytsmixer_forward_pass(phytsmixer_module: PhyTSMixerModule) -> None:
    x_past = torch.rand(1, PhyTSMixerConfig.ctxt_seq_len, 2)
    x_future = torch.rand(1, PhyTSMixerConfig.tgt_seq_len, 1)

    batch = dict(
        encoder_input=x_past,
        decoder_input=torch.cat([x_future, x_future], dim=-1),
    )

    y = phytsmixer_module(batch)

    assert y["z"].shape == (1, PhyTSMixerConfig.tgt_seq_len, 3)
