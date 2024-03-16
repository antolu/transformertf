from __future__ import annotations

import pytest
import torch


from transformertf.nn import ResampleNorm


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(1, 200, 4)


def test_resample_norm(sample: torch.Tensor) -> None:
    model = ResampleNorm(4, 8)

    output = model(sample)

    assert output.shape[:1] == sample.shape[:1]
    assert output.shape[2] == 8


def test_resample_norm_trainable_add_false(sample: torch.Tensor) -> None:
    model = ResampleNorm(4, 8, trainable_add=False)

    output = model(sample)

    assert output.shape[:1] == sample.shape[:1]
    assert output.shape[2] == 8
