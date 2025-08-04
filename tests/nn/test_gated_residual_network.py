"""Tests for Gated Residual Network (GRN) implementation."""

from __future__ import annotations

import pytest
import torch

from transformertf.nn import GatedResidualNetwork


def test_grn_creation():
    """Test GRN module creation."""
    grn = GatedResidualNetwork(
        input_dim=32,
        d_hidden=64,
        output_dim=32,
        dropout=0.1,
    )

    assert grn.input_dim == 32
    assert grn.d_hidden == 64
    assert grn.output_dim == 32
    assert grn.dropout.p == 0.1


def test_grn_forward_without_context():
    """Test GRN forward pass without context."""
    grn = GatedResidualNetwork(
        input_dim=16,
        d_hidden=32,
        output_dim=16,
        dropout=0.1,
    )

    input_tensor = torch.randn(4, 10, 16)
    output = grn(input_tensor)

    assert output.shape == (4, 10, 16)
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_grn_forward_with_context():
    """Test GRN forward pass with context."""
    grn = GatedResidualNetwork(
        input_dim=16,
        d_hidden=32,
        output_dim=16,
        context_dim=8,
        dropout=0.1,
    )

    input_tensor = torch.randn(4, 10, 16)
    context = torch.randn(4, 10, 8)
    output = grn(input_tensor, context)

    assert output.shape == (4, 10, 16)
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_grn_different_dimensions():
    """Test GRN with different input/output dimensions."""
    grn = GatedResidualNetwork(
        input_dim=8,
        d_hidden=16,
        output_dim=12,
        dropout=0.0,
    )

    input_tensor = torch.randn(2, 8)
    output = grn(input_tensor)

    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("projection", ["none", "interpolate", "linear"])
def test_grn_projection_types(projection):
    """Test GRN with different projection types."""
    grn = GatedResidualNetwork(
        input_dim=16,
        d_hidden=32,
        output_dim=16,
        dropout=0.1,
        projection=projection,
    )

    input_tensor = torch.randn(4, 16)
    output = grn(input_tensor)

    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()


def test_grn_gradient_flow():
    """Test gradient flow through GRN."""
    grn = GatedResidualNetwork(
        input_dim=8,
        d_hidden=16,
        output_dim=8,
        dropout=0.1,
    )

    input_tensor = torch.randn(2, 8, requires_grad=True)
    output = grn(input_tensor)

    loss = output.sum()
    loss.backward()

    # Check gradients
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()

    # Check GRN parameter gradients
    for param in grn.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_grn_deterministic():
    """Test GRN deterministic behavior when dropout=0."""
    grn = GatedResidualNetwork(
        input_dim=16,
        d_hidden=32,
        output_dim=16,
        dropout=0.0,
    )
    grn.eval()

    input_tensor = torch.randn(3, 16)

    output1 = grn(input_tensor)
    output2 = grn(input_tensor)

    assert torch.allclose(output1, output2)


def test_grn_with_activation():
    """Test GRN with different activation functions."""
    grn = GatedResidualNetwork(
        input_dim=16,
        d_hidden=32,
        output_dim=16,
        dropout=0.1,
        activation="relu",
    )

    input_tensor = torch.randn(4, 16)
    output = grn(input_tensor)

    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()


def test_grn_training_vs_eval():
    """Test GRN behavior in training vs eval mode."""
    grn = GatedResidualNetwork(
        input_dim=16,
        d_hidden=32,
        output_dim=16,
        dropout=0.5,
    )

    input_tensor = torch.randn(4, 16)

    # Training mode
    grn.train()
    output_train = grn(input_tensor)

    # Eval mode
    grn.eval()
    output_eval = grn(input_tensor)

    assert output_train.shape == output_eval.shape
    assert torch.isfinite(output_train).all()
    assert torch.isfinite(output_eval).all()


def test_grn_context_broadcasting():
    """Test GRN with context broadcasting."""
    grn = GatedResidualNetwork(
        input_dim=8,
        d_hidden=16,
        output_dim=8,
        context_dim=4,
        dropout=0.1,
    )

    input_tensor = torch.randn(2, 10, 8)
    context = torch.randn(2, 10, 4)  # Same shape as input for first two dims

    output = grn(input_tensor, context)

    assert output.shape == (2, 10, 8)
    assert torch.isfinite(output).all()


def test_grn_initialization():
    """Test GRN parameter initialization."""
    grn = GatedResidualNetwork(
        input_dim=32,
        d_hidden=64,
        output_dim=32,
        dropout=0.1,
    )

    # Check that parameters are initialized properly
    for _name, param in grn.named_parameters():
        assert param is not None
        assert torch.isfinite(param).all()

        # Check reasonable initialization ranges
        assert param.abs().max() < 10.0
