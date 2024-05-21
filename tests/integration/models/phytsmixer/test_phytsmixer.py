"""
This module contains complete tests for the PhyLSTM model.
"""

from __future__ import annotations

import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.phytsmixer import PhyTSMixer


def test_phytsmixer_forward_pass(
    phytsmixer_module: PhyTSMixer,
    encoder_decoder_datamodule: EncoderDecoderDataModule,
) -> None:
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()

    dataloader = encoder_decoder_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    phytsmixer_module.on_train_start()
    phytsmixer_module.on_train_epoch_start()

    with torch.no_grad():
        losses = phytsmixer_module.training_step(batch, 0)

    for key in ("loss", "loss1", "loss2", "loss3", "loss4", "loss5"):
        assert key in losses

    phytsmixer_module.on_train_epoch_end()
    phytsmixer_module.on_train_end()

    # validation
    dataloader = encoder_decoder_datamodule.val_dataloader()

    batch = next(iter(dataloader))

    phytsmixer_module.on_validation_start()
    phytsmixer_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = phytsmixer_module.validation_step(batch, 0)

    phytsmixer_module.on_validation_epoch_end()
    phytsmixer_module.on_validation_end()

    for key in (
        "loss",
        "loss1",
        "loss2",
        "loss3",
        "loss4",
        "loss5",
        "output",
    ):
        assert key in outputs
