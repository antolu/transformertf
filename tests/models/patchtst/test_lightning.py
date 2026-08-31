from __future__ import annotations

import pytest
import torch

from transformertf.models.patchtst import PatchTST

BATCH = 2
CTXT = 64
TGT = 20
N_PAST = 4
N_FUTURE = 3


@pytest.fixture(scope="module")
def small_module():
    return PatchTST(
        num_future_features=N_FUTURE,
        ctxt_seq_len=CTXT,
        tgt_seq_len=TGT,
        patch_len=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        d_lstm=48,
        num_lstm_layers=2,
        dropout=0.0,
    )


def _make_batch(
    ctxt: int = CTXT,
    tgt: int = TGT,
    n_past: int = N_PAST,
    n_future: int = N_FUTURE,
    batch: int = BATCH,
) -> dict:
    return {
        "encoder_input": torch.randn(batch, ctxt, n_past),
        "decoder_input": torch.randn(batch, tgt, n_future),
        "target": torch.randn(batch, tgt, 1),
        "encoder_lengths": torch.full((batch, 1), ctxt, dtype=torch.long),
        "decoder_lengths": torch.full((batch, 1), tgt, dtype=torch.long),
    }


def test_construction(small_module):
    assert small_module is not None
    assert small_module.hparams["ctxt_seq_len"] == CTXT
    assert small_module.hparams["tgt_seq_len"] == TGT
    assert small_module.hparams["num_future_features"] == N_FUTURE


def test_forward_output_shape(small_module):
    batch = _make_batch()
    out = small_module(batch)
    assert "output" in out
    assert out["output"].shape == (BATCH, TGT, 1)


def test_training_step(small_module):
    batch = _make_batch()
    result = small_module.training_step(batch, 0)
    assert "loss" in result
    assert torch.isfinite(result["loss"])


def test_validation_step(small_module):
    batch = _make_batch()
    result = small_module.validation_step(batch, 0)
    assert "loss" in result
    assert torch.isfinite(result["loss"])


def test_predict_step(small_module):
    batch = _make_batch()
    result = small_module.predict_step(batch, 0)
    assert "output" in result
    assert "point_prediction" in result
    assert result["point_prediction"].shape == (BATCH, TGT, 1)


def test_hparams_saved(small_module):
    assert small_module.hparams["d_model"] == 32
    assert small_module.hparams["patch_len"] == 16
    assert small_module.hparams["num_layers"] == 2
    assert "criterion" not in small_module.hparams


def test_importable_from_models():
    import importlib

    models = importlib.import_module("transformertf.models")
    assert hasattr(models, "PatchTST")
    assert hasattr(models, "PatchTSTModel")
