from __future__ import annotations

import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.xtft_conv import (
    xTFTConv,
)


def test_xtft_conv_forward_pass_simple(
    xtft_conv_module: xTFTConv,
) -> None:
    x_past = torch.rand(
        1,
        100,
        2,
    )
    x_future = torch.rand(
        1,
        50,
        2,
    )

    batch = {
        "encoder_input": x_past,
        "decoder_input": x_future,
        "encoder_lengths": torch.tensor([[100.0]]),
        "decoder_lengths": torch.tensor([[50.0]]),
        "encoder_mask": torch.ones_like(x_past),
        "decoder_mask": torch.ones_like(x_future),
    }

    with torch.no_grad():
        y = xtft_conv_module(batch)["output"]

    assert y.shape[:2] == x_future.shape[:2]


def test_xtft_conv_forward_pass(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    xtft_conv_module: xTFTConv,
) -> None:
    # hack to remove last 10 values of the val dataset
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()
    encoder_decoder_datamodule.hparams["min_ctxt_seq_len"] = 50
    encoder_decoder_datamodule.hparams["min_tgt_seq_len"] = 25
    encoder_decoder_datamodule.hparams["randomize_seq_len"] = True
    encoder_decoder_datamodule.hparams["batch_size"] = 4
    encoder_decoder_datamodule._val_df[0] = encoder_decoder_datamodule._val_df[0].iloc[  # noqa: SLF001
        :-10
    ]

    dataloader = encoder_decoder_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    xtft_conv_module.on_train_start()
    xtft_conv_module.on_train_epoch_start()

    with torch.no_grad():
        losses = xtft_conv_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    xtft_conv_module.on_train_epoch_end()
    xtft_conv_module.on_train_end()

    # validation
    dataloader = encoder_decoder_datamodule.val_dataloader()

    batch = next(iter(dataloader))
    last_batch = list(dataloader)[-1]

    xtft_conv_module.on_validation_start()
    xtft_conv_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = xtft_conv_module.validation_step(batch, 0)
        last_outputs = xtft_conv_module.validation_step(last_batch, 0)

    xtft_conv_module.on_validation_epoch_end()
    xtft_conv_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs

    for key in (
        "loss",
        "output",
    ):
        assert key in last_outputs
        assert not torch.isnan(last_outputs[key]).any()
