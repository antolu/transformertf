from __future__ import annotations


import pytest
import numpy as np


@pytest.fixture(scope="module")
def x_data() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture(scope="module")
def y_data() -> np.ndarray:
    return np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])


@pytest.fixture(scope="module")
def x_data_2d() -> np.ndarray:
    return np.arange(20).reshape((10, 2))


@pytest.fixture(scope="module")
def y_data_2d() -> np.ndarray:
    return np.arange(20, 40).reshape((10, 2))
