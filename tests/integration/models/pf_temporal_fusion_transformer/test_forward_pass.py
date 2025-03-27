from __future__ import annotations

import typing

import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.pf_tft import (
    PFTemporalFusionTransformer,
)


def test_tft_forward_pass_simple(
    tft_module: PFTemporalFusionTransformer,
    tft_module_config: dict[str, typing.Any],
) -> None:
    x_past = torch.rand(
        1,
        100,
        2,
    )
    x_future = torch.rand(
        1,
        20,
        2,
    )

    batch = {
        "encoder_input": x_past,
        "decoder_input": x_future,
        "encoder_lengths": torch.tensor([[100]]),
        "decoder_lengths": torch.tensor([[20]]),
        "encoder_mask": torch.ones_like(x_past),
        "decoder_mask": torch.ones_like(x_future),
    }

    with torch.no_grad():
        y = tft_module(batch)["output"]

    assert y.shape[:2] == x_future.shape[:2]


def test_tft_forward_pass(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    tft_module: PFTemporalFusionTransformer,
) -> None:
    # hack to remove last 10 values of the val dataset
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()
    encoder_decoder_datamodule.hparams["min_ctxt_seq_len"] = 50
    encoder_decoder_datamodule.hparams["min_tgt_seq_len"] = 25
    encoder_decoder_datamodule.hparams["randomize_seq_len"] = True
    encoder_decoder_datamodule._val_df[0] = encoder_decoder_datamodule._val_df[0].iloc[  # noqa: SLF001
        :-10
    ]

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
    last_batch = list(dataloader)[-1]

    tft_module.on_validation_start()
    tft_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = tft_module.validation_step(batch, 0)
        last_outputs = tft_module.validation_step(last_batch, 0)

    tft_module.on_validation_epoch_end()
    tft_module.on_validation_end()

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
