from __future__ import annotations


import torch
from transformertf.models.tsmixer import FeatureMixer


def test_feature_mixer(sample: torch.Tensor) -> None:
    tm = FeatureMixer(
        num_features=3,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
    )

    out = tm(sample)

    assert sample.shape == out.shape


def test_feature_mixer_out_features(sample: torch.Tensor) -> None:
    tm = FeatureMixer(
        num_features=3,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        out_num_features=2,
    )

    out = tm(sample)

    assert out.shape == (*sample.shape[:2], 2)
