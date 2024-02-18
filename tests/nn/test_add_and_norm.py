from __future__ import annotations


import pytest
import torch

from transformertf.nn import AddAndNorm


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(1, 128, 100)


def test_add_and_norm(sample: torch.Tensor) -> None:
    model = AddAndNorm(100)

    output = model(sample, sample)

    assert model is not None
    assert output.shape == sample.shape
