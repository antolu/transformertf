from __future__ import annotations

from transformertf.models.transformer_v2 import TransformerV2Model


def test_create_transformer_v2() -> None:
    model = TransformerV2Model(num_features=1, seq_len=100)

    assert model is not None
