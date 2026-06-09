from __future__ import annotations

import pytest
import torch

BATCH = 4
CTXT = 64  # must be divisible by patch_len=16
TGT = 32
N_COVARIATES = 2  # num_past_covariates (excl. target)


def test_timexer_model_construction():
    from transformertf.models.timexer import TimeXerModel

    model = TimeXerModel(
        ctxt_seq_len=CTXT,
        tgt_seq_len=TGT,
        num_past_covariates=N_COVARIATES,
        d_model=32,
        num_heads=4,
        num_layers=2,
        patch_len=16,
        d_ff=64,
    )
    assert model is not None


@pytest.fixture(scope="module")
def small_model():
    from transformertf.models.timexer import TimeXerModel

    return TimeXerModel(
        ctxt_seq_len=CTXT,
        tgt_seq_len=TGT,
        num_past_covariates=N_COVARIATES,
        d_model=32,
        num_heads=4,
        num_layers=2,
        patch_len=16,
        d_ff=64,
        dropout=0.0,
    )


def test_timexer_model_output_shape(small_model):
    x_enc = torch.randn(BATCH, CTXT, 1)
    x_ex = torch.randn(BATCH, CTXT + TGT, N_COVARIATES)
    out = small_model(x_enc, x_ex)
    assert out.shape == (BATCH, TGT, 1)


def test_timexer_model_output_finite(small_model):
    x_enc = torch.randn(BATCH, CTXT, 1)
    x_ex = torch.randn(BATCH, CTXT + TGT, N_COVARIATES)
    out = small_model(x_enc, x_ex)
    assert torch.isfinite(out).all()


def test_timexer_model_gradient_flow():
    from transformertf.models.timexer import TimeXerModel

    model = TimeXerModel(
        ctxt_seq_len=CTXT,
        tgt_seq_len=TGT,
        num_past_covariates=N_COVARIATES,
        d_model=32,
        num_heads=4,
        num_layers=2,
        patch_len=16,
        d_ff=64,
        dropout=0.0,
    )
    x_enc = torch.randn(BATCH, CTXT, 1)
    x_ex = torch.randn(BATCH, CTXT + TGT, N_COVARIATES)
    loss = model(x_enc, x_ex).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_timexer_model_patch_len_not_divisor_raises():
    from transformertf.models.timexer import TimeXerModel

    with pytest.raises(ValueError, match="divisible"):
        TimeXerModel(
            ctxt_seq_len=65,  # not divisible by patch_len=16
            tgt_seq_len=TGT,
            num_past_covariates=N_COVARIATES,
            patch_len=16,
        )


def test_timexer_model_deterministic_eval():
    from transformertf.models.timexer import TimeXerModel

    model = TimeXerModel(
        ctxt_seq_len=CTXT,
        tgt_seq_len=TGT,
        num_past_covariates=N_COVARIATES,
        d_model=32,
        num_heads=4,
        num_layers=2,
        patch_len=16,
        d_ff=64,
        dropout=0.0,
    )
    model.eval()
    x_enc = torch.randn(BATCH, CTXT, 1)
    x_ex = torch.randn(BATCH, CTXT + TGT, N_COVARIATES)
    with torch.no_grad():
        out1 = model(x_enc, x_ex)
        out2 = model(x_enc, x_ex)
    assert torch.allclose(out1, out2, atol=1e-6)
