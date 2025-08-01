"""Benchmark tests for model performance."""

from __future__ import annotations

import pytest
import torch

from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformerModel,
)

from ..test_utils import assert_tensor_finite


@pytest.mark.benchmark
def test_forward_pass_performance():
    """Benchmark forward pass performance."""
    model = TemporalFusionTransformerModel(
        num_past_features=3,
        num_future_features=3,
        ctxt_seq_len=100,
        tgt_seq_len=50,
        num_lstm_layers=2,
        d_model=64,
        num_heads=8,
        output_dim=1,
        hidden_continuous_dim=32,
    )

    past = torch.randn(8, 100, 3)
    future = torch.randn(8, 50, 3)

    # Simple performance test without benchmark fixture
    result = model(past, future)
    assert_tensor_finite(result["output"])


@pytest.mark.benchmark
def test_attention_performance():
    """Benchmark attention mechanism performance."""
    from transformertf.nn import InterpretableMultiHeadAttention

    attention = InterpretableMultiHeadAttention(
        d_model=256,
        num_heads=8,
        dropout=0.1,
    )

    input_tensor = torch.randn(16, 200, 256)

    # Simple performance test without benchmark fixture
    result = attention(input_tensor, input_tensor, input_tensor)
    assert_tensor_finite(result)


@pytest.mark.benchmark
def test_large_batch_performance():
    """Benchmark performance with large batches."""
    model = TemporalFusionTransformerModel(
        num_past_features=2,
        num_future_features=2,
        ctxt_seq_len=50,
        tgt_seq_len=25,
        num_lstm_layers=1,
        d_model=32,
        num_heads=4,
        output_dim=1,
        hidden_continuous_dim=16,
    )

    past = torch.randn(64, 50, 2)
    future = torch.randn(64, 25, 2)

    # Simple performance test without benchmark fixture
    result = model(past, future)
    assert_tensor_finite(result["output"])
