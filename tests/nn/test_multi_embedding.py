from __future__ import annotations

import pytest
import torch

from transformertf.nn import MultiEmbedding


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.randint(0, 10, (1, 200, 2))


def test_multi_embedding(sample: torch.Tensor) -> None:
    model = MultiEmbedding([(10, 8), (10, 8)])

    output = model(sample)

    assert output.shape[:2] == sample.shape[:2]
    assert output.shape[2] == 8
    assert output.shape[3] == 2
