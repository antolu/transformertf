from __future__ import annotations

from transformertf.models.transformerencoder import TransformerEncoder


def test_create_transformer_encoder() -> None:
    model = TransformerEncoder(num_features=1, seq_len=100)

    assert model is not None
