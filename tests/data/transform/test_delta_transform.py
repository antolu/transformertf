from __future__ import annotations

import numpy as np
import torch

from transformertf.data.transform import DeltaTransform


def test_delta_transform_forward_with_tensor() -> None:
    transform = DeltaTransform()
    y = torch.tensor([1.0, 2.0, 4.0, 7.0])
    result = transform.forward(y)
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(result, expected)


def test_delta_transform_forward_with_empty_tensor() -> None:
    transform = DeltaTransform()
    y = torch.tensor([])
    result = transform.forward(y)
    expected = torch.tensor([])
    assert torch.allclose(result, expected)


def test_delta_transform_transform_with_tensor() -> None:
    transform = DeltaTransform()
    x = torch.tensor([1.0, 2.0, 4.0, 7.0])
    result = transform.transform(x)
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(result, expected)


def test_delta_transform_transform_with_numpy() -> None:
    transform = DeltaTransform()
    x = np.array([1.0, 2.0, 4.0, 7.0])
    result = transform.transform(x)
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    assert torch.allclose(result, expected)


def test_delta_transform_inverse_transform_with_tensor() -> None:
    transform = DeltaTransform()
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    result = transform.inverse_transform(y)
    expected = torch.tensor([0.0, 1.0, 3.0, 6.0])
    assert torch.allclose(result, expected)


def test_delta_transform_inverse_transform_with_numpy() -> None:
    transform = DeltaTransform()
    y = np.array([0.0, 1.0, 2.0, 3.0])
    result = transform.inverse_transform(y)
    expected = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=torch.float64)
    assert torch.allclose(result, expected)


def test_delta_transform_forward_with_single_element_tensor() -> None:
    transform = DeltaTransform()
    y = torch.tensor([5.0])
    result = transform.forward(y)
    expected = torch.tensor([0.0])
    assert torch.allclose(result, expected)


def test_delta_transform_inverse_transform_with_single_element_tensor() -> None:
    transform = DeltaTransform()
    y = torch.tensor([5.0])
    result = transform.inverse_transform(y)
    expected = torch.tensor([5.0])
    assert torch.allclose(result, expected)


def test_delta_transform_inverse_transform_with_multiple_elements_tensor() -> None:
    transform = DeltaTransform()
    y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    result = transform.inverse_transform(y)
    expected = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0])
    assert torch.allclose(result, expected)


def test_delta_transform_inverse_transform_with_nonzero_first_element() -> None:
    transform = DeltaTransform()
    y = torch.tensor([1.0, 2.0, 4.0, 7.0])
    result = transform.inverse_transform(transform.transform(y))

    assert not torch.allclose(y, result)
    assert torch.allclose(y, result + y[0])
