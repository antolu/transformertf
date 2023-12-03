from __future__ import annotations


import torch
from ._commons import NUM_FEATURES
from transformertf.models.tsmixer import MixerBlock


def test_mixer_block(sample: torch.Tensor) -> None:
    mixer = MixerBlock(
        num_features=NUM_FEATURES,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
    )

    out = mixer(sample)

    assert sample.shape == out.shape


def test_mixer_block_out_features(sample: torch.Tensor) -> None:
    tm = MixerBlock(
        num_features=NUM_FEATURES,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        out_num_features=2,
    )

    out = tm(sample)

    assert out.shape == (*sample.shape[:2], 2)
