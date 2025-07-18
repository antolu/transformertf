"""
Tests for the LogTransform class.
"""

from __future__ import annotations

import numpy as np
import torch

from transformertf.data.transform import Log1pTransform, LogTransform


def test_log_transform_torch() -> None:
    transform = LogTransform()

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.log(x)

    x_hat = transform.transform(x)
    assert torch.allclose(y, torch.log(x))

    x_hat = transform.inverse_transform(x_hat)
    assert torch.allclose(x, x_hat)


def test_log_transform_numpy() -> None:
    transform = LogTransform()

    # Test with numpy arrays
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = np.log(x)

    x_hat = transform.transform(x)
    assert np.allclose(y, np.log(x))
    assert isinstance(x_hat, torch.Tensor)

    x_hat = transform.inverse_transform(x_hat)
    assert np.allclose(x, x_hat)
    assert isinstance(x_hat, torch.Tensor)


def test_log_transform_fitted() -> None:
    transform = LogTransform()

    # verify with sklearn
    import sklearn.utils.validation

    sklearn.utils.validation.check_is_fitted(transform)


def test_log_transform_str_repr() -> None:
    """assert that the __str__ and __repr__ methods do not raise an exception"""
    transform = LogTransform()

    _ = str(transform)
    _ = repr(transform)


def test_log1p_transform_torch() -> None:
    transform = Log1pTransform()

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.log1p(x)

    x_hat = transform.transform(x)
    assert torch.allclose(y, torch.log1p(x))

    x_hat = transform.inverse_transform(x_hat)
    assert torch.allclose(x, x_hat)


def test_log1p_transform_numpy() -> None:
    transform = Log1pTransform()

    # Test with numpy arrays
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = np.log1p(x)

    x_hat = transform.transform(x)
    assert np.allclose(y, np.log1p(x))
    assert isinstance(x_hat, torch.Tensor)

    x_hat = transform.inverse_transform(x_hat)
    assert np.allclose(x, x_hat)
    assert isinstance(x_hat, torch.Tensor)


def test_log1p_transform_fitted() -> None:
    transform = Log1pTransform()

    # verify with sklearn
    import sklearn.utils.validation

    sklearn.utils.validation.check_is_fitted(transform)


def test_log1p_transform_str_repr() -> None:
    """assert that the __str__ and __repr__ methods do not raise an exception"""
    transform = Log1pTransform()

    _ = str(transform)
    _ = repr(transform)


def test_log_transform_edge_cases():
    """Test LogTransform with edge cases."""
    transform = LogTransform()

    # Test with very small positive values
    x_small = torch.tensor([1e-8, 1e-6, 1e-4])
    y_small = transform.transform(x_small)
    assert torch.isfinite(y_small).all()

    # Test inverse transform
    x_recovered = transform.inverse_transform(y_small)
    assert torch.allclose(x_small, x_recovered, atol=1e-6)


def test_log_transform_gradient_flow():
    """Test gradient flow through LogTransform."""
    transform = LogTransform()

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = transform.transform(x)

    loss = y.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_log1p_transform_edge_cases():
    """Test Log1pTransform with edge cases."""
    transform = Log1pTransform()

    # Test with zero values
    x_zero = torch.tensor([0.0, 1.0, 2.0])
    y_zero = transform.transform(x_zero)
    assert torch.isfinite(y_zero).all()

    # Test with negative values close to -1
    x_neg = torch.tensor([-0.5, -0.1, 0.5])
    y_neg = transform.transform(x_neg)
    assert torch.isfinite(y_neg).all()

    # Test inverse transform
    x_recovered = transform.inverse_transform(y_neg)
    assert torch.allclose(x_neg, x_recovered, atol=1e-6)


def test_log1p_transform_gradient_flow():
    """Test gradient flow through Log1pTransform."""
    transform = Log1pTransform()

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = transform.transform(x)

    loss = y.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_log_transform_different_shapes():
    """Test LogTransform with different tensor shapes."""
    transform = LogTransform()

    # Test 2D tensor
    x_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_2d = transform.transform(x_2d)
    assert y_2d.shape == (2, 2)

    # Test 3D tensor
    x_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    y_3d = transform.transform(x_3d)
    assert y_3d.shape == (1, 2, 2)


def test_log1p_transform_different_shapes():
    """Test Log1pTransform with different tensor shapes."""
    transform = Log1pTransform()

    # Test 2D tensor
    x_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_2d = transform.transform(x_2d)
    assert y_2d.shape == (2, 2)

    # Test 3D tensor
    x_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    y_3d = transform.transform(x_3d)
    assert y_3d.shape == (1, 2, 2)
