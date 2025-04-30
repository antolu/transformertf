from __future__ import annotations

import pytest
import torch

from transformertf.data.transform import MinMaxScaler


@pytest.fixture
def min_max_scaler() -> MinMaxScaler:
    return MinMaxScaler(num_features_=2)


def test_min_max_scaler_not_fitted(min_max_scaler: MinMaxScaler) -> None:
    assert not min_max_scaler.__sklearn_is_fitted__()


def test_min_max_scaler_forward(min_max_scaler: MinMaxScaler) -> None:
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    min_max_scaler.fit(y)
    result = min_max_scaler.forward(y)
    expected = (y - torch.tensor([1.0, 2.0])) / (
        torch.tensor([3.0, 4.0]) - torch.tensor([1.0, 2.0])
    )
    assert torch.allclose(result, expected)


def test_min_max_scaler_fit(min_max_scaler: MinMaxScaler) -> None:
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    min_max_scaler.fit(y)
    assert min_max_scaler.__sklearn_is_fitted__()


def test_min_max_scaler_transform(min_max_scaler: MinMaxScaler) -> None:
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    min_max_scaler.fit(y)
    result = min_max_scaler.transform(y)
    expected = (y - torch.tensor([1.0, 2.0])) / (
        torch.tensor([3.0, 4.0]) - torch.tensor([1.0, 2.0])
    )
    assert torch.allclose(result, expected)


def test_min_max_scaler_inverse_transform(min_max_scaler: MinMaxScaler) -> None:
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    min_max_scaler.fit(y)
    y_scaled = min_max_scaler.transform(y)
    y_unscaled = min_max_scaler.inverse_transform(y_scaled)
    assert torch.allclose(y, y_unscaled)


def test_min_max_scaler_state_dict(min_max_scaler: MinMaxScaler) -> None:
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    min_max_scaler.fit(y)
    state_dict = min_max_scaler.state_dict()

    for key in ("min_", "max_", "num_features_", "data_min_", "data_max_", "frozen_"):
        assert key in state_dict
