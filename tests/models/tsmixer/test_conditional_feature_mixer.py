from __future__ import annotations

import torch

from transformertf.models.tsmixer import ConditionalMixerBlock

from .conftest import NUM_FEATURES, NUM_STATIC_FEATURES, SEQ_LEN


def test_conditional_feature_mixer(
    sample: torch.Tensor, static_covariates: torch.Tensor
) -> None:
    tm = ConditionalMixerBlock(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        num_static_features=NUM_STATIC_FEATURES,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        out_num_features=NUM_FEATURES,
    )

    out = tm(sample, static_covariates)

    assert sample.shape == out.shape


def test_feature_mixer_out_features(
    sample: torch.Tensor, static_covariates: torch.Tensor
) -> None:
    tm = ConditionalMixerBlock(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        num_static_features=NUM_STATIC_FEATURES,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        out_num_features=NUM_FEATURES - 1,
    )

    out = tm(sample, static_covariates)

    assert out.shape == (*sample.shape[:2], NUM_FEATURES - 1)
