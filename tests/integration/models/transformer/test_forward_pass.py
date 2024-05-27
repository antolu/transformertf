from __future__ import annotations

import typing

import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.transformer import (
    VanillaTransformer,
)


def test_transformer_forward_pass_simple(
    transformer_module: VanillaTransformer,
    transformer_module_config: dict[str, typing.Any],
) -> None:
    x_past = torch.rand(
        1,
        transformer_module_config["ctxt_seq_len"],
        2,
    )
    x_future = torch.rand(
        1,
        transformer_module_config["tgt_seq_len"],
        2,
    )

    batch = {
        "encoder_input": x_past,
        "decoder_input": x_future,
    }

    with torch.no_grad():
        y = transformer_module(batch)

    assert y.shape[:2] == x_future.shape[:2]


def test_transformer_v2_forward_pass_point(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    transformer_module: VanillaTransformer,
) -> None:
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()

    dataloader = encoder_decoder_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    transformer_module.on_train_start()
    transformer_module.on_train_epoch_start()

    with torch.no_grad():
        losses = transformer_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    transformer_module.on_train_epoch_end()
    transformer_module.on_train_end()

    # validation
    dataloader = encoder_decoder_datamodule.val_dataloader()

    batch = next(iter(dataloader))

    transformer_module.on_validation_start()
    transformer_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = transformer_module.validation_step(batch, 0)

    transformer_module.on_validation_epoch_end()
    transformer_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs


def test_transformer_v2_forward_pass_delta(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    transformer_module: VanillaTransformer,
) -> None:
    transformer_module.hparams["prediction_type"] = "delta"
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()

    dataloader = encoder_decoder_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    transformer_module.on_train_start()
    transformer_module.on_train_epoch_start()

    with torch.no_grad():
        losses = transformer_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    transformer_module.on_train_epoch_end()
    transformer_module.on_train_end()

    # validation
    dataloader = encoder_decoder_datamodule.val_dataloader()

    batch = next(iter(dataloader))

    transformer_module.on_validation_start()
    transformer_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = transformer_module.validation_step(batch, 0)

    transformer_module.on_validation_epoch_end()
    transformer_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs
