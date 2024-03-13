from __future__ import annotations

import numpy as np
import torch

from transformertf.data.transform import DivideByXTransform


def test_divide_by_x_transform_torch() -> None:
    transform = DivideByXTransform()

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

    y_div_x = transform.transform(x, y)
    assert torch.allclose(y_div_x, y / x)

    y_mult_x = transform.inverse_transform(x, y_div_x)
    assert torch.allclose(y_mult_x, y)


def test_divide_by_x_transform_numpy() -> None:
    transform = DivideByXTransform()

    # Test with numpy arrays
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

    y_div_x = transform.transform(x, y)
    assert np.allclose(y_div_x, y / x)
    assert isinstance(y_div_x, torch.Tensor)

    y_mult_x = transform.inverse_transform(x, y_div_x)
    assert np.allclose(y_mult_x, y)
    assert isinstance(y_mult_x, torch.Tensor)


def test_divide_by_x_transform_fitted() -> None:
    transform = DivideByXTransform()

    # verify with sklearn
    import sklearn.utils.validation

    sklearn.utils.validation.check_is_fitted(transform)
