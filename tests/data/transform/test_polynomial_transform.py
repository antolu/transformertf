from __future__ import annotations

import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from sklearn.utils.validation import check_is_fitted

from transformertf.data import FixedPolynomialTransform, PolynomialTransform

from ...strategies import polynomial_coefficients_strategy, tensor_strategy


@pytest.fixture
def x() -> torch.Tensor:
    return torch.ones(2) * 2


def test_polynomial_transform_zero_degree(x: torch.Tensor) -> None:
    transform = PolynomialTransform(degree=0, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data = torch.ones(1)

    y = transform(x)
    assert y.shape == (2,)

    assert torch.eq(y, torch.ones(2)).all()


def test_polynomial_transform_one_degree(x: torch.Tensor) -> None:
    transform = PolynomialTransform(degree=1, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data -= 2.0
    transform.weights.data = torch.ones(1) * 3.0

    y = transform(x)
    assert y.shape == (2,)

    assert torch.eq(y, torch.ones(2) * 4.0).all()


def test_polynomial_transform_two_degree(x: torch.Tensor) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data -= 2.0
    transform.weights.data = torch.ones(2) * 3.0

    y = transform(x)
    assert y.shape == (2,)

    assert torch.eq(y, torch.ones(2) * 16.0).all()


def test_polynomial_transform_zero_degree_derivative() -> None:
    transform = PolynomialTransform(degree=0, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data = torch.ones(1)

    with pytest.raises(ValueError):  # noqa: PT011
        transform.get_derivative()


def test_polynomial_transform_one_degree_derivative() -> None:
    transform = PolynomialTransform(degree=1, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data -= 2.0
    transform.weights.data = torch.ones(1) * 3.0

    derivative = transform.get_derivative()
    assert derivative.degree == 0
    assert derivative.weights.shape == (0,)
    assert derivative.bias.shape == (1,)

    assert derivative.bias == 3.0


def test_polynomial_transform_two_degree_derivative() -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data -= 2.0
    transform.weights.data = torch.ones(2) * 3.0

    derivative = transform.get_derivative()
    assert derivative.degree == 1
    assert derivative.weights.shape == (1,)
    assert derivative.bias.shape == (1,)

    assert derivative.bias == 3.0
    assert torch.eq(derivative.weights, torch.ones(1) * 6.0).all()


def test_polynomial_transform_three_degree_derivative() -> None:
    transform = PolynomialTransform(degree=3, num_iterations=10)
    transform._reset_parameters()

    transform.bias.data -= 2.0
    transform.weights.data = torch.ones(3) * 3.0

    derivative = transform.get_derivative()
    assert derivative.degree == 2
    assert derivative.weights.shape == (2,)
    assert derivative.bias.shape == (1,)

    assert derivative.bias == 3.0
    assert torch.eq(derivative.weights, torch.tensor([6.0, 9.0])).all()


def test_polynomial_transform_fit(
    df: pd.DataFrame, current_key: str, field_key: str
) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)

    transform.fit(df[current_key].to_numpy(), df[field_key].to_numpy())

    check_is_fitted(transform)


def test_polynomial_transform_transform(
    df: pd.DataFrame, current_key: str, field_key: str
) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)

    transform.fit(df[current_key].to_numpy(), df[field_key].to_numpy())

    check_is_fitted(transform)

    transformed = transform.transform(
        df[current_key].to_numpy(), df[field_key].to_numpy()
    )

    assert transformed.shape == (len(df),)


def test_polynomial_transform_inverse_transform(
    df: pd.DataFrame, current_key: str, field_key: str
) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)

    transform.fit(df[current_key].to_numpy(), df[field_key].to_numpy())

    check_is_fitted(transform)

    transformed = transform.transform(
        df[current_key].to_numpy(), df[field_key].to_numpy()
    )

    assert transformed.shape == (len(df),)

    inverse_transformed = transform.inverse_transform(
        df[current_key].to_numpy(), transformed
    )

    assert inverse_transformed.shape == (len(df),)


def test_fixed_polynomial_transform_line() -> None:
    transform = FixedPolynomialTransform(degree=1, weights=[2.0], bias=1.0)

    x = torch.ones(2) * 2

    y = transform(x)
    assert y.shape == (2,)

    assert torch.eq(y, torch.ones(2) * 5.0).all()


def test_polynomial_transform_str_repr() -> None:
    """assert that the __str__ and __repr__ methods do not raise an exception"""
    transform = PolynomialTransform(degree=2, num_iterations=10)
    transform.fit(torch.ones(2), torch.ones(2))

    _ = str(transform)
    _ = repr(transform)


@pytest.mark.property
@given(
    coeffs=polynomial_coefficients_strategy(degree=3),
    input_tensor=tensor_strategy(shape=(10,), min_value=-5.0, max_value=5.0),
)
@settings(max_examples=15, deadline=None)
def test_polynomial_transform_properties(coeffs, input_tensor):
    """Property-based test for polynomial transform properties."""
    transform = PolynomialTransform(degree=len(coeffs) - 1, num_iterations=100)

    # Manually set coefficients for testing forward pass
    transform.bias.data = torch.tensor([coeffs[0]])
    if len(coeffs) > 1:
        transform.weights.data = torch.tensor(coeffs[1:])

    output = transform(input_tensor)

    # Property: output should be finite for finite inputs
    assert torch.isfinite(output).all()

    # Property: output should have same shape as input
    assert output.shape == input_tensor.shape


def test_polynomial_transform_edge_cases():
    """Test polynomial transform with edge cases."""
    transform = PolynomialTransform(degree=2, num_iterations=10)

    # Test with zero input
    zero_input = torch.zeros(3)
    output = transform(zero_input)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()

    # Test with negative input
    neg_input = torch.tensor([-1.0, -2.0, -3.0])
    output = transform(neg_input)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()

    # Test with single element
    single_input = torch.tensor([5.0])
    output = transform(single_input)
    assert output.shape == (1,)
    assert torch.isfinite(output).all()


def test_polynomial_transform_gradient_flow():
    """Test that gradients flow through polynomial transform."""
    transform = PolynomialTransform(degree=2, num_iterations=10)

    input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    output = transform(input_tensor)
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()

    # Check that transform parameters have gradients
    assert transform.bias.grad is not None
    assert transform.weights.grad is not None
    assert torch.isfinite(transform.bias.grad).all()
    assert torch.isfinite(transform.weights.grad).all()


def test_polynomial_transform_different_degrees():
    """Test polynomial transform with different degrees."""
    input_tensor = torch.tensor([1.0, 2.0, 3.0])

    for degree in [0, 1, 2, 3, 4]:
        transform = PolynomialTransform(degree=degree, num_iterations=10)
        output = transform(input_tensor)

        assert output.shape == (3,)
        assert torch.isfinite(output).all()

        # Check parameter shapes
        assert transform.bias.shape == (1,)
        assert transform.weights.shape == (degree,)


def test_fixed_polynomial_transform_properties():
    """Test fixed polynomial transform properties."""
    # Test different degrees
    for degree in [1, 2, 3]:
        weights = [1.0] * degree
        bias = 0.5
        transform = FixedPolynomialTransform(degree=degree, weights=weights, bias=bias)

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = transform(input_tensor)

        assert output.shape == (3,)
        assert torch.isfinite(output).all()


def test_polynomial_transform_consistency():
    """Test that polynomial transforms are consistent across calls."""
    transform = PolynomialTransform(degree=2, num_iterations=10)

    # Set fixed weights
    transform.bias.data = torch.tensor([1.0])
    transform.weights.data = torch.tensor([2.0, 3.0])

    input_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Run multiple times
    output1 = transform(input_tensor)
    output2 = transform(input_tensor)

    # Should be identical
    assert torch.allclose(output1, output2)
