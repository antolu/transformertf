from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.data.transform import DiscreteFunctionTransform


@pytest.fixture
def numpy_arrays() -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([4.0, 5.0, 6.0])
    return xs, ys


@pytest.fixture
def torch_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.tensor([1.0, 2.0, 3.0])
    ys = torch.tensor([4.0, 5.0, 6.0])
    return xs, ys


def test_discrete_fn_init_with_numpy_arrays(
    numpy_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    xs, ys = numpy_arrays
    transform = DiscreteFunctionTransform(xs, ys)

    np.testing.assert_array_equal(transform.xs_, xs)
    np.testing.assert_array_equal(transform.ys_, ys)


def test_discrete_fn_init_with_torch_tensors(
    torch_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    xs, ys = torch_tensors
    transform = DiscreteFunctionTransform(xs, ys)

    np.testing.assert_array_equal(transform.xs_, xs.numpy())
    np.testing.assert_array_equal(transform.ys_, ys.numpy())


def test_discrete_fn_forward_with_numpy_array(
    numpy_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    xs, ys = numpy_arrays
    transform = DiscreteFunctionTransform(xs, ys)

    x = np.array([1.5, 2.5])
    result = transform.forward(x)
    expected = torch.tensor([4.5, 5.5], dtype=torch.float64)

    torch.testing.assert_close(result, expected)


def test_discrete_fn_forward_with_torch_tensor(
    torch_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    xs, ys = torch_tensors
    transform = DiscreteFunctionTransform(xs, ys)

    x = torch.tensor([1.5, 2.5])
    result = transform.forward(x)
    expected = torch.tensor([4.5, 5.5], dtype=torch.float64)

    torch.testing.assert_close(result, expected)


def test_discrete_fn_transform_with_numpy_array(
    numpy_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    xs, ys = numpy_arrays
    transform = DiscreteFunctionTransform(xs, ys)

    x = np.array([1.5, 2.5])
    y = np.array([5.0, 6.0])

    result = transform.transform(x, y)
    expected = torch.tensor([0.5, 0.5], dtype=torch.float64)

    torch.testing.assert_close(result, expected)


def test_discrete_fn_transform_with_torch_tensor(
    torch_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    xs, ys = torch_tensors
    transform = DiscreteFunctionTransform(xs, ys)

    x = torch.tensor([1.5, 2.5])
    y = torch.tensor([5.0, 6.0])

    result = transform.transform(x, y)
    expected = torch.tensor([0.5, 0.5], dtype=torch.float64)

    torch.testing.assert_close(result, expected)


def test_discrete_fn_inverse_transform_with_numpy_array(
    numpy_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    xs, ys = numpy_arrays
    transform = DiscreteFunctionTransform(xs, ys)

    x = np.array([1.5, 2.5])
    y = np.array([5.0, 6.0])

    result = transform.inverse_transform(x, y)
    expected = torch.tensor([9.5, 11.5], dtype=torch.float64)

    torch.testing.assert_close(result, expected)


def test_discrete_fn_inverse_transform_with_torch_tensor(
    torch_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    xs, ys = torch_tensors
    transform = DiscreteFunctionTransform(xs, ys)

    x = torch.tensor([1.5, 2.5])
    y = torch.tensor([5.0, 6.0])

    result = transform.inverse_transform(x, y)
    expected = torch.tensor([9.5, 11.5], dtype=torch.float64)

    torch.testing.assert_close(result, expected)


def test_discrete_fn_from_csv(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "data.csv"
    data = pd.DataFrame({"xs": [1.0, 2.0, 3.0], "ys": [4.0, 5.0, 6.0]})
    data.to_csv(csv_path, index=False)

    transform = DiscreteFunctionTransform.from_csv(csv_path)

    np.testing.assert_array_equal(transform.xs_, data["xs"].to_numpy())
    np.testing.assert_array_equal(transform.ys_, data["ys"].to_numpy())

    transform = DiscreteFunctionTransform(csv_path)

    np.testing.assert_array_equal(transform.xs_, data["xs"].to_numpy())
    np.testing.assert_array_equal(transform.ys_, data["ys"].to_numpy())

    transform = DiscreteFunctionTransform(str(csv_path))

    np.testing.assert_array_equal(transform.xs_, data["xs"].to_numpy())
    np.testing.assert_array_equal(transform.ys_, data["ys"].to_numpy())
