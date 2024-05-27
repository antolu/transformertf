from __future__ import annotations

import typing

import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)


def test_transformer_v2_forward_pass_simple(
    tft_module: TemporalFusionTransformer,
    tft_module_config: dict[str, typing.Any],
) -> None:
    x_past = torch.rand(
        1,
        tft_module_config["ctxt_seq_len"],
        2,
    )
    x_future = torch.rand(
        1,
        tft_module_config["tgt_seq_len"],
        2,
    )

    batch = {
        "encoder_input": x_past,
        "decoder_input": x_future,
        "encoder_lengths": torch.tensor([[1.0]]),
    }

    with torch.no_grad():
        y = tft_module(batch)["output"]

    assert y.shape[:2] == x_future.shape[:2]


def test_transformer_v2_forward_pass(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    tft_module: TemporalFusionTransformer,
) -> None:
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()

    dataloader = encoder_decoder_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    tft_module.on_train_start()
    tft_module.on_train_epoch_start()

    with torch.no_grad():
        losses = tft_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    tft_module.on_train_epoch_end()
    tft_module.on_train_end()

    # validation
    dataloader = encoder_decoder_datamodule.val_dataloader()

    batch = next(iter(dataloader))

    tft_module.on_validation_start()
    tft_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = tft_module.validation_step(batch, 0)

    tft_module.on_validation_epoch_end()
    tft_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs
