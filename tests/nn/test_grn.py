from __future__ import annotations


import pytest
import torch

from transformertf.nn import GatedResidualNetwork


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.rand(1, 128, 100)


def test_gated_residual_network(sample: torch.Tensor) -> None:
    model = GatedResidualNetwork(
        input_dim=100,
        output_dim=100,
        context_dim=100,
        dropout=0.1,
        activation="elu",
    )

    output = model(sample, sample)

    assert model is not None
    assert output.shape == sample.shape


def test_gated_residual_network_with_context(sample: torch.Tensor) -> None:
    model = GatedResidualNetwork(
        input_dim=100,
        output_dim=100,
        context_dim=50,
        dropout=0.1,
        activation="elu",
    )

    context = torch.rand(1, 50)

    output = model(sample, context)

    assert model is not None
    assert output.shape == sample.shape
