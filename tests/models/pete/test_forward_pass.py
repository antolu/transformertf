from __future__ import annotations

import pytest
import torch

from transformertf.models.pete import PETE


@pytest.fixture
def pete_model() -> PETE:
    return PETE(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=100,
        d_selection=64,
        d_model=80,
        num_encoder_heads=4,
        num_layers=2,
        dropout=0.1,
    )


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor]:
    return {
        "encoder_input": torch.randn(4, 100, 2),
        "decoder_input": torch.randn(4, 50, 2),
        "target": torch.randn(4, 50, 1),
        "encoder_lengths": torch.ones(4, 1),
        "decoder_lengths": torch.ones(4, 1),
        "encoder_mask": torch.ones(4, 100, 2),
        "decoder_mask": torch.ones(4, 50, 2),
    }


def test_forward_pass(pete_model: PETE, sample_batch: dict[str, torch.Tensor]) -> None:
    output = pete_model.validation_step(sample_batch, batch_idx=0)

    assert "output" in output
