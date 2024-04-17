from __future__ import annotations

import pytest
import torch

from transformertf.nn import InterpretableMultiHeadAttention


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(2, 200, 4)


def test_interpretable_multi_head_attention(sample: torch.Tensor) -> None:
    model = InterpretableMultiHeadAttention(
        n_dim_model=4,
        n_heads=2,
        dropout=0.1,
    )

    output, _attn = model(sample, sample, sample, return_attn=True)

    assert output.shape == sample.shape
