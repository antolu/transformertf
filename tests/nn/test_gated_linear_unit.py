"""Tests for Gated Linear Unit (GLU) implementation."""

from __future__ import annotations

import pytest
import torch

from transformertf.nn import GatedLinearUnit


def test_glu_creation():
    """Test GLU module creation."""
    glu = GatedLinearUnit(input_dim=32, dropout=0.1)

    assert glu.input_dim == 32
    assert glu.dropout.p == 0.1
    assert glu.output_dim == 32


def test_glu_forward():
    """Test GLU forward pass."""
    glu = GatedLinearUnit(input_dim=16, dropout=0.1)

    input_tensor = torch.randn(4, 10, 16)
    output = glu(input_tensor)

    assert output.shape == (4, 10, 16)
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_glu_different_input_shapes():
    """Test GLU with different input shapes."""
    glu = GatedLinearUnit(input_dim=8, dropout=0.0)

    # Test 2D input
    input_2d = torch.randn(5, 8)
    output_2d = glu(input_2d)
    assert output_2d.shape == (5, 8)

    # Test 3D input
    input_3d = torch.randn(2, 10, 8)
    output_3d = glu(input_3d)
    assert output_3d.shape == (2, 10, 8)


def test_glu_activation():
    """Test GLU activation function."""
    glu = GatedLinearUnit(input_dim=4, dropout=0.0)

    # Test with known values
    input_tensor = torch.ones(1, 4)
    output = glu(input_tensor)

    # GLU should apply sigmoid gating
    assert torch.isfinite(output).all()
    assert output.shape == (1, 4)


def test_glu_deterministic():
    """Test GLU deterministic behavior when dropout=0."""
    glu = GatedLinearUnit(input_dim=16, dropout=0.0)
    glu.eval()

    input_tensor = torch.randn(3, 16)

    output1 = glu(input_tensor)
    output2 = glu(input_tensor)

    assert torch.allclose(output1, output2)


def test_glu_gradient_flow():
    """Test gradient flow through GLU."""
    glu = GatedLinearUnit(input_dim=8, dropout=0.1)

    input_tensor = torch.randn(2, 8, requires_grad=True)
    output = glu(input_tensor)

    loss = output.sum()
    loss.backward()

    # Check gradients
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()

    # Check GLU parameter gradients
    for param in glu.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
def test_glu_dropout_rates(dropout):
    """Test GLU with different dropout rates."""
    glu = GatedLinearUnit(input_dim=16, dropout=dropout)

    input_tensor = torch.randn(4, 10, 16)
    output = glu(input_tensor)

    assert output.shape == (4, 10, 16)
    assert torch.isfinite(output).all()


def test_glu_training_vs_eval():
    """Test GLU behavior in training vs eval mode."""
    glu = GatedLinearUnit(input_dim=16, dropout=0.5)

    input_tensor = torch.randn(4, 16)

    # Training mode
    glu.train()
    output_train = glu(input_tensor)

    # Eval mode
    glu.eval()
    output_eval = glu(input_tensor)

    assert output_train.shape == output_eval.shape
    assert torch.isfinite(output_train).all()
    assert torch.isfinite(output_eval).all()


def test_glu_initialization():
    """Test GLU parameter initialization."""
    glu = GatedLinearUnit(input_dim=32, dropout=0.1)

    # Check that parameters are initialized properly
    for _name, param in glu.named_parameters():
        assert param is not None
        assert torch.isfinite(param).all()

        # Check reasonable initialization ranges
        assert param.abs().max() < 10.0
