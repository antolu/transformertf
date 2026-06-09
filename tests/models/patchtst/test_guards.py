from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from transformertf.main import _check_patchtst_constraints
from transformertf.models.patchtst import PatchTST


def _make_model() -> PatchTST:
    return PatchTST(
        num_past_covariates=4,
        num_future_covariates=3,
        ctxt_seq_len=64,
        tgt_seq_len=20,
        patch_len=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        lstm_hidden=48,
        lstm_num_layers=2,
        dropout=0.0,
    )


def _make_dm(randomize=False, time_column=None, n_past=4, n_future=3):
    dm = MagicMock()
    dm.hparams = {
        "randomize_seq_len": randomize,
        "time_column": time_column,
    }
    dm.num_past_known_covariates = n_past
    dm.num_future_known_covariates = n_future
    return dm


def test_guard_randomize_seq_len():
    model = _make_model()
    dm = _make_dm(randomize=True)
    with pytest.raises(ValueError, match="randomize_seq_len"):
        _check_patchtst_constraints(model, dm)


def test_guard_time_column():
    model = _make_model()
    dm = _make_dm(time_column="timestamp")
    with pytest.raises(ValueError, match="time_column"):
        _check_patchtst_constraints(model, dm)


def test_guard_covariate_mismatch():
    model = _make_model()
    dm = _make_dm(n_past=5, n_future=3)
    with pytest.raises(ValueError, match="covariate"):
        _check_patchtst_constraints(model, dm)


def test_guard_passes_valid_config():
    model = _make_model()
    dm = _make_dm()
    _check_patchtst_constraints(model, dm)


def test_guard_skips_non_patchtst():
    from transformertf.models.timexer import TimeXer

    model = TimeXer(
        num_past_covariates=3,
        ctxt_seq_len=64,
        tgt_seq_len=32,
        d_model=32,
        num_heads=4,
        num_layers=2,
        patch_len=16,
        d_ff=64,
        dropout=0.0,
    )
    dm = _make_dm(randomize=True)
    _check_patchtst_constraints(model, dm)
