from __future__ import annotations

import torch

from transformertf.models.tsmixer import MixerBlock

from .conftest import NUM_FEATURES, SEQ_LEN


def test_mixer_block(sample: torch.Tensor) -> None:
    mixer = MixerBlock(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        d_fc=64,
        dropout=0.1,
        norm="batch",
    )

    out = mixer(sample)

    assert sample.shape == out.shape


def test_mixer_block_out_features(sample: torch.Tensor) -> None:
    tm = MixerBlock(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        d_fc=64,
        dropout=0.1,
        norm="batch",
        out_num_features=2,
    )

    out = tm(sample)

    assert out.shape == (*sample.shape[:2], 2)
