from __future__ import annotations

import pytest
import torch

from transformertf.nn import AddNorm


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(1, 128, 100)


def test_add_norm(sample: torch.Tensor) -> None:
    model = AddNorm(100)

    output = model(sample, sample)

    assert model is not None
    assert output.shape == sample.shape


def test_add_norm_trainable_false(sample: torch.Tensor) -> None:
    model = AddNorm(100, trainable_add=False)

    output = model(sample, sample)

    assert model is not None
    assert output.shape == sample.shape
