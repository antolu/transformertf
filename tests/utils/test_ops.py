from __future__ import annotations


from transformertf.utils import ops
import numpy as np
import torch
import pytest


@pytest.fixture(scope="module")
def np_arr() -> np.ndarray:
    return np.ones(10)


@pytest.fixture(scope="module")
def torch_arr() -> torch.Tensor:
    return torch.ones(10)


@pytest.fixture(scope="module")
def list_of_np_arrs() -> list[np.ndarray]:
    return [np.ones(10) for _ in range(10)]


@pytest.fixture(scope="module")
def list_of_torch_arrs() -> list[torch.Tensor]:
    return [torch.ones(10) for _ in range(10)]


@pytest.fixture(scope="module")
def tuple_of_np_arrs() -> tuple[np.ndarray, ...]:
    return tuple([np.ones(10) for _ in range(10)])


@pytest.fixture(scope="module")
def tuple_of_torch_arrs() -> tuple[torch.Tensor, ...]:
    return tuple([torch.ones(10) for _ in range(10)])


@pytest.fixture(scope="module")
def list_of_dicts_numpy() -> list[dict[str, np.ndarray]]:
    return [
        {"a": np.ones(10), "b": np.ones(10)},
        {"a": np.ones(10), "b": np.ones(10)},
    ]


@pytest.fixture(scope="module")
def list_of_dicts_torch() -> list[dict[str, torch.Tensor]]:
    return [
        {"a": torch.ones(10), "b": torch.ones(10)},
        {"a": torch.ones(10), "b": torch.ones(10)},
    ]


@pytest.mark.parametrize(
    "arr,shape",
    [
        ("list_of_np_arrs", (100,)),
        ("list_of_torch_arrs", (100,)),
        ("tuple_of_np_arrs", (100,)),
        ("tuple_of_torch_arrs", (100,)),
    ],
)
def test_concatenate_sequences(
    arr: pytest.fixture, shape: tuple[int, ...], request: pytest.FixtureRequest
) -> None:
    """
    Test truncating a sequence of arrays.
    """
    arr = request.getfixturevalue(arr)
    arr = ops.concatenate(arr)

    assert arr.shape == shape


@pytest.mark.parametrize(
    "arr,shape",
    [
        ("list_of_dicts_numpy", (20,)),
        ("list_of_dicts_torch", (20,)),
    ],
)
def test_concatenate_sequences_of_dicts(
    arr: pytest.fixture, shape: tuple[int, ...], request: pytest.FixtureRequest
) -> None:
    """
    Test truncating a sequence of arrays.
    """
    arr = request.getfixturevalue(arr)
    arr = ops.concatenate(arr)

    assert arr["a"].shape == shape
    assert arr["b"].shape == shape
