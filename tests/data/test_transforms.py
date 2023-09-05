from __future__ import annotations
from ..conftest import CURRENT, FIELD
import pandas as pd
from sklearn.utils.validation import check_is_fitted
import pytest
import torch

from transformertf.data import PolynomialTransform


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

    with pytest.raises(ValueError):
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


def test_polynomial_transform_fit(df: pd.DataFrame) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)

    transform.fit(df[CURRENT].values, df[FIELD].values)

    check_is_fitted(transform)


def test_polynomial_transform_transform(df: pd.DataFrame) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)

    transform.fit(df[CURRENT].values, df[FIELD].values)

    check_is_fitted(transform)

    transformed = transform.transform(df[CURRENT].values, df[FIELD].values)

    assert transformed.shape == (len(df),)


def test_polynomial_transform_inverse_transform(df: pd.DataFrame) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=10)

    transform.fit(df[CURRENT].values, df[FIELD].values)

    check_is_fitted(transform)

    transformed = transform.transform(df[CURRENT].values, df[FIELD].values)

    assert transformed.shape == (len(df),)

    inverse_transformed = transform.inverse_transform(
        df[CURRENT].values, transformed
    )

    assert inverse_transformed.shape == (len(df),)
