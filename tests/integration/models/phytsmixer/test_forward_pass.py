from __future__ import annotations

import pytest
import torch

from transformertf.models.phytsmixer import PhyTSMixer, PhyTSMixerConfig


@pytest.fixture(scope="module")
def phytsmixer_module() -> PhyTSMixer:
    return PhyTSMixer.from_config(
        PhyTSMixerConfig(
            num_blocks=2,
            fc_dim=16,
            input_columns=["I_meas_A"],
            target_column="B_meas_T",
        )
    )


def test_phytsmixer_forward_pass(phytsmixer_module: PhyTSMixer) -> None:
    x_past = torch.rand(1, PhyTSMixerConfig.ctxt_seq_len, 2)
    x_future = torch.rand(1, PhyTSMixerConfig.tgt_seq_len, 1)

    batch = {
        "encoder_input": x_past,
        "decoder_input": torch.cat([x_future, x_future], dim=-1),
    }

    with torch.no_grad():
        y = phytsmixer_module(batch)

    assert y["z"].shape == (1, PhyTSMixerConfig.tgt_seq_len, 3)
