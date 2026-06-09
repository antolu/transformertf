from __future__ import annotations

import pytest
import torch

B = 2
CTXT = 64  # must be divisible by patch_len=16
TGT = 20
C_ENC = 4  # past covariates incl. target
C_DEC = 3  # future covariates = C_ENC - 1


@pytest.fixture(scope="module")
def small_model():
    from transformertf.models.patchtst._model import PatchTSTModel

    return PatchTSTModel(
        num_future_covariates=C_DEC,
        ctxt_seq_len=CTXT,
        patch_len=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        lstm_hidden=48,
        lstm_num_layers=2,
        dropout=0.0,
        use_revin=True,
    )


def test_patchtst_model_construction(small_model):
    assert small_model is not None


def test_patchtst_model_output_shape(small_model):
    enc = torch.randn(B, CTXT, C_ENC)
    dec = torch.randn(B, TGT, C_DEC)
    out = small_model(enc, dec)
    assert out.shape == (B, TGT, 1)


def test_patchtst_model_output_finite(small_model):
    enc = torch.randn(B, CTXT, C_ENC)
    dec = torch.randn(B, TGT, C_DEC)
    out = small_model(enc, dec)
    assert torch.isfinite(out).all()


def test_patchtst_model_gradient_flow():
    from transformertf.models.patchtst._model import PatchTSTModel

    model = PatchTSTModel(
        num_future_covariates=C_DEC,
        ctxt_seq_len=CTXT,
        patch_len=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        lstm_hidden=48,
        lstm_num_layers=2,
        dropout=0.0,
        use_revin=True,
    )
    enc = torch.randn(B, CTXT, C_ENC)
    dec = torch.randn(B, TGT, C_DEC)
    loss = model(enc, dec).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad: {name}"


def test_patchtst_model_patch_len_guard():
    from transformertf.models.patchtst._model import PatchTSTModel

    with pytest.raises(ValueError, match="divisible"):
        PatchTSTModel(
            num_future_covariates=C_DEC,
            ctxt_seq_len=65,  # not divisible by 16
            patch_len=16,
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            lstm_hidden=48,
            lstm_num_layers=2,
            dropout=0.0,
            use_revin=True,
        )


def test_patchtst_model_deterministic_eval(small_model):
    small_model.eval()
    enc = torch.randn(B, CTXT, C_ENC)
    dec = torch.randn(B, TGT, C_DEC)
    with torch.no_grad():
        out1 = small_model(enc, dec)
        out2 = small_model(enc, dec)
    assert torch.allclose(out1, out2, atol=1e-6)
    small_model.train()
