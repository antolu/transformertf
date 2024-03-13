from __future__ import annotations

import torch

from transformertf.models.tsmixer import ConditionalMixerBlock

from .conftest import NUM_FEATURES, NUM_STATIC_FEATURES, SEQ_LEN


def test_conditional_mixer_block(
    sample: torch.Tensor, static_covariates: torch.Tensor
) -> None:
    mixer = ConditionalMixerBlock(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        num_static_features=NUM_STATIC_FEATURES,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
    )

    out = mixer(sample, static_covariates)

    assert sample.shape == out.shape


def test_conditional_mixer_block_out_features(
    sample: torch.Tensor, static_covariates: torch.Tensor
) -> None:
    tm = ConditionalMixerBlock(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        num_static_features=NUM_STATIC_FEATURES,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        out_num_features=2,
    )

    out = tm(sample, static_covariates)

    assert out.shape == (*sample.shape[:2], 2)
