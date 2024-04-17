"""
This module contains complete tests for the PhyLSTM model.
"""

from __future__ import annotations

import pytest
import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.phytsmixer import PhyTSMixerConfig, PhyTSMixerModule

from ....conftest import CURRENT, DF_PATH


@pytest.fixture(scope="module")
def config() -> PhyTSMixerConfig:
    return PhyTSMixerConfig(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        num_workers=0,
        target_depends_on=CURRENT,
        input_columns=["I_meas_A"],
        target_column="B_meas_T",
    )


@pytest.fixture()
def phytsmixer_module(config: PhyTSMixerConfig) -> PhyTSMixerModule:
    return PhyTSMixerModule.from_config(config)


@pytest.fixture()
def datamodule(config: PhyTSMixerConfig) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule.from_parquet(config)


def test_phytsmixer_forward_pass(
    phytsmixer_module: PhyTSMixerModule,
    datamodule: EncoderDecoderDataModule,
) -> None:
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

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
    dataloader = datamodule.val_dataloader()

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
