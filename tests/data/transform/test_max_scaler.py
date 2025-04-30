from __future__ import annotations

import torch

from transformertf.data.transform import MaxScaler


def test_max_scaler_not_fitted() -> None:
    scaler = MaxScaler(num_features_=2)
    assert not scaler.__sklearn_is_fitted__()


def test_max_scaler_forward() -> None:
    scaler = MaxScaler(num_features_=2)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(y)
    result = scaler.forward(y)
    expected = y / torch.tensor([3.0, 4.0])
    assert torch.allclose(result, expected)


def test_max_scaler_fit() -> None:
    scaler = MaxScaler(num_features_=2)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(y)
    assert scaler.__sklearn_is_fitted__()


def test_max_scaler_transform() -> None:
    scaler = MaxScaler(num_features_=2)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(y)
    result = scaler.transform(y)
    expected = y / torch.tensor([3.0, 4.0])
    assert torch.allclose(result, expected)


def test_max_scaler_inverse_transform() -> None:
    scaler = MaxScaler(num_features_=2)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(y)
    y_scaled = scaler.transform(y)
    y_unscaled = scaler.inverse_transform(y_scaled)
    assert torch.allclose(y, y_unscaled)
