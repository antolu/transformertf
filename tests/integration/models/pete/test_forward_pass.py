from __future__ import annotations

import pytest

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.pete import PETE


@pytest.fixture
def pete_model() -> PETE:
    return PETE(
        ctxt_seq_len=100,
        n_dim_selection=64,
        n_dim_model=80,
        n_enc_heads=4,
        num_layers=2,
        dropout=0.1,
    )


def test_forward_pass(
    pete_model: PETE, encoder_decoder_datamodule: EncoderDecoderDataModule
) -> None:
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()

    batch = next(iter(encoder_decoder_datamodule.train_dataloader()))

    output = pete_model.validation_step(batch, batch_idx=0)

    assert "output" in output
