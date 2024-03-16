from __future__ import annotations

import pytest
import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformerConfig,
    TemporalFusionTransformerModule,
)

from ....conftest import CURRENT, FIELD, DF_PATH


@pytest.fixture(scope="module")
def config() -> TemporalFusionTransformerConfig:
    return TemporalFusionTransformerConfig(
        n_dim_model=16,
        num_lstm_layers=1,
        num_heads=4,
        train_dataset=[DF_PATH],
        val_dataset=[DF_PATH],
        batch_size=4,
        num_workers=0,
        target_depends_on=CURRENT,
        input_columns=[CURRENT],
        target_column=FIELD,
    )


@pytest.fixture(scope="module")
def tft_module(
    config: TemporalFusionTransformerConfig,
) -> TemporalFusionTransformerModule:
    return TemporalFusionTransformerModule.from_config(config)


@pytest.fixture
def datamodule(
    config: TemporalFusionTransformerConfig,
) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule.from_parquet(config)


def test_transformer_v2_forward_pass_simple(
    config: TemporalFusionTransformerConfig,
    tft_module: TemporalFusionTransformerModule,
) -> None:
    x_past = torch.rand(
        1,
        config.ctxt_seq_len,
        2,
    )
    x_future = torch.rand(
        1,
        config.tgt_seq_len,
        2,
    )

    batch = dict(
        encoder_input=x_past,
        decoder_input=x_future,
    )

    with torch.no_grad():
        y = tft_module(batch)["output"]

    assert y.shape[:2] == x_future.shape[:2]


def test_transformer_v2_forward_pass(
    datamodule: EncoderDecoderDataModule,
    tft_module: TemporalFusionTransformerModule,
) -> None:
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))

    tft_module.on_train_start()
    tft_module.on_train_epoch_start()

    with torch.no_grad():
        losses = tft_module.training_step(batch, 0)

    for key in ("loss", "loss_MSE"):
        assert key in losses

    tft_module.on_train_epoch_end()
    tft_module.on_train_end()

    # validation
    dataloader = datamodule.val_dataloader()

    batch = next(iter(dataloader))

    tft_module.on_validation_start()
    tft_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = tft_module.validation_step(batch, 0)

    tft_module.on_validation_epoch_end()
    tft_module.on_validation_end()

    for key in (
        "loss",
        "loss_MSE",
        "output",
    ):
        assert key in outputs
