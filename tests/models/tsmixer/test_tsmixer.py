from __future__ import annotations

import torch

from transformertf.models.tsmixer import BasicTSMixer, TSMixer

from .conftest import BATCH_SIZE, NUM_FEATURES, NUM_STATIC_FEATURES, SEQ_LEN

OUT_SEQ_LEN = 12


def test_tsmixer(
    sample: torch.Tensor,
    future_covariates: torch.Tensor,
    static_covariates: torch.Tensor,
) -> None:
    mixer = TSMixer(
        num_feat=NUM_FEATURES,
        num_future_feat=NUM_FEATURES,
        num_static_real_feat=NUM_STATIC_FEATURES,
        num_blocks=3,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        activation="relu",
        seq_len=SEQ_LEN,
        out_seq_len=OUT_SEQ_LEN,
    )

    out = mixer(sample, future_covariates, static_covariates)

    assert out.shape == (BATCH_SIZE, OUT_SEQ_LEN, NUM_FEATURES)


# def test_tsmixer_headless(sample: torch.Tensor) -> None:
#     mixer = TSMixer(
#         num_features=NUM_FEATURES,
#         num_blocks=3,
#         fc_dim=64,
#         dropout=0.1,
#         norm="batch",
#         activation="relu",
#         seq_len=SEQ_LEN,
#         out_seq_len="headless",
#     )
#
#     out = mixer(sample)
#
#     assert out.shape == sample.shape


def test_tsmixer_target_slice(
    sample: torch.Tensor,
    future_covariates: torch.Tensor,
    static_covariates: torch.Tensor,
) -> None:
    mixer = BasicTSMixer(
        num_features=NUM_FEATURES,
        num_blocks=3,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        activation="relu",
        seq_len=SEQ_LEN,
        out_seq_len=OUT_SEQ_LEN,
    )

    out = mixer(sample, target_slice=1)

    assert out.shape == (BATCH_SIZE, OUT_SEQ_LEN, 1)
