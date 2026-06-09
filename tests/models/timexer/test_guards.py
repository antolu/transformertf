from __future__ import annotations

import pytest


def test_guard_randomize_seq_len():
    from unittest.mock import MagicMock

    from transformertf.main import _check_timexer_constraints
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
    datamodule = MagicMock()
    datamodule.hparams = {"randomize_seq_len": True}
    datamodule.num_past_known_covariates = 3
    datamodule.num_future_known_covariates = 2

    with pytest.raises(ValueError, match="randomize_seq_len"):
        _check_timexer_constraints(model, datamodule)


def test_guard_covariate_mismatch():
    from unittest.mock import MagicMock

    from transformertf.main import _check_timexer_constraints
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
    datamodule = MagicMock()
    datamodule.hparams = {"randomize_seq_len": False}
    datamodule.num_past_known_covariates = 4
    datamodule.num_future_known_covariates = 2

    with pytest.raises(ValueError, match="covariate"):
        _check_timexer_constraints(model, datamodule)


def test_guard_passes_valid_config():
    from unittest.mock import MagicMock

    from transformertf.main import _check_timexer_constraints
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
    datamodule = MagicMock()
    datamodule.hparams = {"randomize_seq_len": False}
    datamodule.num_past_known_covariates = 3
    datamodule.num_future_known_covariates = 2

    _check_timexer_constraints(model, datamodule)  # must not raise
