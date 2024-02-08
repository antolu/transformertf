from __future__ import annotations

import pytest
import torch

from transformertf.models.transformerencoder import (
    TransformerEncoderModule,
    TransformerEncoderConfig,
)


@pytest.fixture(scope="module")
def transformer_encoder_module() -> TransformerEncoderModule:
    return TransformerEncoderModule.from_config(
        TransformerEncoderConfig(
            input_columns=["I_meas_A"], target_column="B_meas_T"
        )
    )


def test_transformer_encoder_forward_pass(
    transformer_encoder_module: TransformerEncoderModule,
) -> None:
    x_past = torch.rand(
        1,
        TransformerEncoderConfig.ctxt_seq_len * 2
        + TransformerEncoderConfig.tgt_seq_len,
        1,
    )
    x_future = torch.rand(
        1,
        TransformerEncoderConfig.ctxt_seq_len
        + TransformerEncoderConfig.tgt_seq_len,
        1,
    )

    batch = dict(
        encoder_input=x_past,
    )

    with torch.no_grad():
        y = transformer_encoder_module(batch)

    assert y.shape[:2] == x_future.shape[:2]
