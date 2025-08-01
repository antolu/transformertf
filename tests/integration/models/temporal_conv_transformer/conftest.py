from __future__ import annotations

import typing

import pytest

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.temporal_conv_transformer import TemporalConvTransformer


@pytest.fixture
def tct_module_config(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
) -> dict[str, typing.Any]:
    """Configuration for TCT module testing."""
    return {
        "num_past_features": encoder_decoder_datamodule.num_past_known_covariates,
        "num_future_features": encoder_decoder_datamodule.num_future_known_covariates,
        "output_dim": 1,
        "d_model": 64,
        "num_heads": 4,
        "compression_factor": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dropout": 0.1,
        "ctxt_seq_len": 400,  # Long enough for compression
        "tgt_seq_len": 100,  # Long enough for compression
    }


@pytest.fixture
def tct_module(tct_module_config: dict[str, typing.Any]) -> TemporalConvTransformer:
    """Create TCT module for testing."""
    # Extract only the model parameters (not sequence length parameters)
    model_params = {
        k: v
        for k, v in tct_module_config.items()
        if k not in ["ctxt_seq_len", "tgt_seq_len"]
    }

    return TemporalConvTransformer(**model_params)
