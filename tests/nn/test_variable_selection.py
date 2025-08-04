from __future__ import annotations

import pytest
import torch

from transformertf.nn import VariableSelection


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(1, 200, 4)


def test_variable_selection(sample: torch.Tensor) -> None:
    model = VariableSelection(4, d_hidden=8, d_model=32)

    output, _weights = model(sample)

    assert model is not None
    assert output.shape[:1] == sample.shape[:1]
    assert output.shape[2] == 32


def test_variable_selection_context(sample: torch.Tensor) -> None:
    model = VariableSelection(4, d_hidden=8, d_model=32, context_size=8)
    context = torch.rand(1, 200, 8)

    output, _weights = model(sample, context)

    assert model is not None
    assert output.shape[:1] == sample.shape[:1]
    assert output.shape[2] == 32
