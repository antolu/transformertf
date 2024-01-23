from __future__ import annotations


import torch
from transformertf.models.tsmixer import TimeMixer
from .conftest import SEQ_LEN


def test_time_mixer(sample: torch.Tensor) -> None:
    tm = TimeMixer(
        input_len=SEQ_LEN,
        dropout=0.1,
        norm="batch",
    )

    out = tm(sample)

    assert sample.shape == out.shape
