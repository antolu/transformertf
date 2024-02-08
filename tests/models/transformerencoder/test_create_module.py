from __future__ import annotations

from transformertf.models.transformerencoder import TransformerEncoder


def test_create_transformer_encoder() -> None:
    model = TransformerEncoder(seq_len=100, out_seq_len=10)

    assert model is not None
