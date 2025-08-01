from __future__ import annotations

import pytest
import torch

from transformertf.nn import AddNorm, GateAddNorm, GatedLinearUnit


@pytest.fixture
def gate_add_norm() -> GateAddNorm:
    return GateAddNorm(input_dim=10, output_dim=20, dropout=0.1, trainable_add=True)


def initialization_correctly_sets_attributes(gate_add_norm: GateAddNorm) -> None:
    assert gate_add_norm.input_dim == 10
    assert gate_add_norm.output_dim == 20
    assert isinstance(gate_add_norm.glu, GatedLinearUnit)
    assert isinstance(gate_add_norm.add_norm, AddNorm)


def forward_pass_returns_correct_shape(gate_add_norm: GateAddNorm) -> None:
    x = torch.randn(32, 10, 50)  # batch_size=32, input_dim=10, num_features=50
    residual = torch.randn(32, 20, 50)  # batch_size=32, output_dim=20, num_features=50

    output = gate_add_norm(x, residual)

    assert output.shape == (32, 20, 50)


def trainable_add_is_correctly_set(gate_add_norm: GateAddNorm) -> None:
    assert gate_add_norm.add_norm.trainable_add is True


def forward_pass_with_mismatched_dimensions_raises_error(
    gate_add_norm: GateAddNorm,
) -> None:
    x = torch.randn(32, 10, 50)  # batch_size=32, input_dim=10, num_features=50
    residual = torch.randn(32, 30, 50)  # batch_size=32, output_dim=30, num_features=50

    with pytest.raises(RuntimeError):
        gate_add_norm(x, residual)
