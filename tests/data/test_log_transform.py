"""
Tests for the LogTransform class.
"""

from __future__ import annotations

import numpy as np
import torch

from transformertf.data.transform import LogTransform


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
    import sklearn.utils.validation  # noqa: PLC0415

    sklearn.utils.validation.check_is_fitted(transform)
