from __future__ import annotations


import torch
from transformertf.models.tsmixer import BatchNorm2D

from ._commons import sample  # noqa: F401


def test_batch_norm_2d(sample: torch.Tensor) -> None:  # noqa: F811
    bn = BatchNorm2D(num_features=1)

    out = bn(sample)

    assert sample.shape == out.shape
