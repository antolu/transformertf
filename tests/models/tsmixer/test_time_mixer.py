from __future__ import annotations


import torch
from transformertf.models.tsmixer import TimeMixer
from .conftest import SEQ_LEN, NUM_FEATURES


def test_time_mixer_batch(sample: torch.Tensor) -> None:
    tm = TimeMixer(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        dropout=0.1,
        norm="batch",
    )

    out = tm(sample)

    assert sample.shape == out.shape


def test_time_mixer_layer_norm(sample: torch.Tensor) -> None:
    tm = TimeMixer(
        input_len=SEQ_LEN,
        num_features=NUM_FEATURES,
        dropout=0.1,
        norm="layer",
    )

    out = tm(sample)

    assert sample.shape == out.shape
