from __future__ import annotations

import torch
import pytest


BATCH_SIZE = 2
NUM_FEATURES = 3
SEQ_LEN = 32

NUM_STATIC_FEATURES = 10
PRED_SEQ_LEN = 12


@pytest.fixture(scope="session")
def sample() -> torch.Tensor:
    return torch.rand((BATCH_SIZE, SEQ_LEN, NUM_FEATURES))


@pytest.fixture(scope="session")
def static_covariates() -> torch.Tensor:
    return torch.rand((BATCH_SIZE, NUM_STATIC_FEATURES))


@pytest.fixture(scope="session")
def future_covariates() -> torch.Tensor:
    return torch.rand((BATCH_SIZE, PRED_SEQ_LEN, NUM_FEATURES))
