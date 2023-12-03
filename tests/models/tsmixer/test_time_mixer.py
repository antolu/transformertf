from __future__ import annotations


import torch
from transformertf.models.tsmixer import TimeMixer


def test_time_mixer(sample: torch.Tensor) -> None:
    tm = TimeMixer(
        num_features=3,
        dropout=0.1,
        norm="batch",
    )

    out = tm(sample)

    assert sample.shape == out.shape
