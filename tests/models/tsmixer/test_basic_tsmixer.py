from __future__ import annotations


import torch
from .conftest import NUM_FEATURES, SEQ_LEN, BATCH_SIZE
from transformertf.models.tsmixer import BasicTSMixer


OUT_SEQ_LEN = 12


def test_basic_tsmixer(sample: torch.Tensor) -> None:
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

    out = mixer(sample)

    assert out.shape == (BATCH_SIZE, OUT_SEQ_LEN, NUM_FEATURES)


def test_basic_tsmixer_headless(sample: torch.Tensor) -> None:
    mixer = BasicTSMixer(
        num_features=NUM_FEATURES,
        num_blocks=3,
        fc_dim=64,
        dropout=0.1,
        norm="batch",
        activation="relu",
        seq_len=SEQ_LEN,
        out_seq_len=None,
    )

    out = mixer(sample)

    assert out.shape == sample.shape


def test_basic_tsmixer_target_slice(sample: torch.Tensor) -> None:
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
