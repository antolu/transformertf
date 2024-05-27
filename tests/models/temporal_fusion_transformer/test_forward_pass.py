from __future__ import annotations

import pytest
import torch

from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformerModel,
)

BATCH_SIZE = 4
PAST_SEQ_LEN = 100
FUTURE_SEQ_LEN = 10
NUM_FEATURES = 2


@pytest.fixture(scope="module")
def past_covariates() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, PAST_SEQ_LEN, NUM_FEATURES)


@pytest.fixture(scope="module")
def future_covariates() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, FUTURE_SEQ_LEN, NUM_FEATURES - 1)


def test_temporal_fusion_transformer_model(
    past_covariates: torch.Tensor, future_covariates: torch.Tensor
) -> None:
    model = TemporalFusionTransformerModel(
        num_past_features=NUM_FEATURES,
        num_future_features=1,
        ctxt_seq_len=PAST_SEQ_LEN,
        tgt_seq_len=FUTURE_SEQ_LEN,
        num_lstm_layers=1,
        n_dim_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=12,
    )

    output = model(past_covariates, future_covariates)

    assert output["output"].shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)
