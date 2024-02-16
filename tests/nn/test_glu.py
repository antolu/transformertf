from __future__ import annotations


import pytest
import torch

from transformertf.nn import GatedLinearUnit


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(1, 128, 100)


def test_gated_linear_unit(sample: torch.Tensor) -> None:
    model = GatedLinearUnit(100)

    output = model(sample)

    assert model is not None
    assert output.shape == sample.shape
