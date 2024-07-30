from __future__ import annotations

import pytest
import torch

from transformertf.data import TimeSeriesSample
from transformertf.models.lstm import LSTM

BATCH_SIZE = 4
SEQ_LEN = 100


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, 1)
    y = torch.randn(BATCH_SIZE, SEQ_LEN, 1)
    return x, y


@pytest.fixture
def sample(batch: tuple[torch.Tensor, torch.Tensor]) -> TimeSeriesSample:
    x, y = batch
    return {
        "input": x,
        "target": y,
        "initial_state": torch.cat((x[:, 0], y[:, 0]), dim=-1),
    }


@pytest.fixture
def lstm_module() -> LSTM:
    module = LSTM(num_features=1, n_dim_model=10, num_layers=1)
    assert module is not None
    return module


def test_forward_pass(
    lstm_module: LSTM, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    with torch.no_grad():
        y_hat = lstm_module(x)
    assert y_hat.shape == y.shape


def test_forward_pass_with_states(
    lstm_module: LSTM, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    with torch.no_grad():
        y_hat, _ = lstm_module(x, return_states=True)
    assert y_hat.shape == y.shape


def test_training_step(lstm_module: LSTM, sample: TimeSeriesSample) -> None:
    with torch.no_grad():
        loss = lstm_module.training_step(sample, batch_idx=0)
    assert loss is not None


def test_validation_step(lstm_module: LSTM, sample: TimeSeriesSample) -> None:
    with torch.no_grad():
        loss = lstm_module.validation_step(sample, batch_idx=0)
    assert loss is not None
    assert isinstance(loss, dict)
    for key in ("loss", "output", "state"):
        assert key in loss
