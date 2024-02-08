from __future__ import annotations
import pytest
import torch


from transformertf.data import TimeSeriesSample
from transformertf.models.lstm import LSTMConfig, LSTMModule


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
def module() -> LSTMModule:
    config = LSTMConfig()
    module = LSTMModule.from_config(config)
    assert module is not None
    return module


def test_forward_pass(
    module: LSTMModule, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    with torch.no_grad():
        y_hat = module(x)
    assert y_hat.shape == y.shape


def test_forward_pass_with_states(
    module: LSTMModule, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    with torch.no_grad():
        y_hat, _ = module(x, return_states=True)
    assert y_hat.shape == y.shape


def test_training_step(module: LSTMModule, sample: TimeSeriesSample) -> None:
    with torch.no_grad():
        loss = module.training_step(sample, batch_idx=0)
    assert loss is not None


def test_validation_step(module: LSTMModule, sample: TimeSeriesSample) -> None:
    with torch.no_grad():
        loss = module.validation_step(sample, batch_idx=0)
    assert loss is not None
    assert isinstance(loss, dict)
    for key in ("loss", "output", "state"):
        assert key in loss
