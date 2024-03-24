from __future__ import annotations


from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)


def test_create_temporal_fusion_transformer() -> None:
    model = TemporalFusionTransformer(
        ctxt_seq_len=10,
        tgt_seq_len=5,
        n_dim_model=8,
        num_heads=2,
        num_lstm_layers=2,
        num_past_covariates=1,
        num_future_covariates=1,
        num_static_cont=1,
        dropout=0.1,
    )

    assert model is not None
