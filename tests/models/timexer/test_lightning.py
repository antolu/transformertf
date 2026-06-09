from __future__ import annotations

import pytest
import torch

from transformertf.models.timexer import TimeXer

BATCH = 4
CTXT = 64
TGT = 32
NUM_PAST_COVARIATES = 3  # includes target, so model gets 2 exogenous channels


@pytest.fixture(scope="module")
def timexer_module():
    return TimeXer(
        num_past_covariates=NUM_PAST_COVARIATES,
        ctxt_seq_len=CTXT,
        tgt_seq_len=TGT,
        d_model=32,
        num_heads=4,
        num_layers=2,
        patch_len=16,
        d_ff=64,
        dropout=0.0,
    )


def _make_batch(
    ctxt: int = CTXT,
    tgt: int = TGT,
    n_past: int = NUM_PAST_COVARIATES,
    n_future: int = NUM_PAST_COVARIATES - 1,
    batch: int = BATCH,
) -> dict:
    return {
        "encoder_input": torch.randn(batch, ctxt, n_past),
        "decoder_input": torch.randn(batch, tgt, n_future),
        "target": torch.randn(batch, tgt, 1),
        "encoder_lengths": torch.full((batch, 1), ctxt, dtype=torch.long),
        "decoder_lengths": torch.full((batch, 1), tgt, dtype=torch.long),
    }


def test_timexer_lightning_construction(timexer_module):
    assert timexer_module is not None
    assert timexer_module.hparams["ctxt_seq_len"] == CTXT
    assert timexer_module.hparams["tgt_seq_len"] == TGT
    assert timexer_module.hparams["num_past_covariates"] == NUM_PAST_COVARIATES


def test_timexer_forward_output_shape(timexer_module):
    batch = _make_batch()
    out = timexer_module(batch)
    assert "output" in out
    assert out["output"].shape == (BATCH, TGT, 1)


def test_timexer_training_step(timexer_module):
    batch = _make_batch()
    result = timexer_module.training_step(batch, 0)
    assert "loss" in result
    assert torch.isfinite(result["loss"])


def test_timexer_validation_step(timexer_module):
    batch = _make_batch()
    result = timexer_module.validation_step(batch, 0)
    assert "loss" in result
    assert torch.isfinite(result["loss"])


def test_timexer_predict_step(timexer_module):
    batch = _make_batch()
    result = timexer_module.predict_step(batch, 0)
    assert "output" in result
    assert "point_prediction" in result
    assert result["point_prediction"].shape == (BATCH, TGT, 1)


def test_timexer_hparams_saved(timexer_module):
    assert timexer_module.hparams["d_model"] == 32
    assert timexer_module.hparams["patch_len"] == 16
    assert timexer_module.hparams["num_layers"] == 2


def test_timexer_importable_from_models():
    import importlib

    models = importlib.import_module("transformertf.models")
    assert hasattr(models, "TimeXer")
