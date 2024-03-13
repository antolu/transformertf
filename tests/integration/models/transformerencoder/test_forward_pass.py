from __future__ import annotations

import pytest
import torch

from transformertf.data import EncoderDataModule
from transformertf.models.transformerencoder import (
    TransformerEncoderConfig,
    TransformerEncoderModule,
)

from ....conftest import CURRENT, DF_PATH


@pytest.fixture(scope="module")
def config() -> TransformerEncoderConfig:
    return TransformerEncoderConfig(
        fc_dim=64,
        n_dim_model=16,
        num_encoder_layers=2,
        num_heads=2,
        train_dataset=[DF_PATH],
        val_dataset=[DF_PATH],
        num_workers=0,
        target_depends_on=CURRENT,
        input_columns=["I_meas_A"],
        target_column="B_meas_T",
    )


@pytest.fixture(scope="module")
def transformer_encoder_module(
    config: TransformerEncoderConfig,
) -> TransformerEncoderModule:
    return TransformerEncoderModule.from_config(config)


@pytest.fixture
def datamodule(config: TransformerEncoderConfig) -> EncoderDataModule:
    return EncoderDataModule.from_parquet(config)


def test_transformer_encoder_forward_pass_simple(
    transformer_encoder_module: TransformerEncoderModule,
) -> None:
    x_past = torch.rand(
        1,
        TransformerEncoderConfig.ctxt_seq_len
        + TransformerEncoderConfig.tgt_seq_len,
        2,
    )
    x_future = torch.rand(
        1,
        TransformerEncoderConfig.ctxt_seq_len
        + TransformerEncoderConfig.tgt_seq_len,
        1,
    )

    batch = dict(
        encoder_input=x_past,
    )

    with torch.no_grad():
        y = transformer_encoder_module(batch)

    assert y.shape[:2] == x_future.shape[:2]


def test_transformer_encoder_forward_pass(
    datamodule: EncoderDataModule,
    transformer_encoder_module: TransformerEncoderModule,
) -> None:
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))

    transformer_encoder_module.on_train_start()
    transformer_encoder_module.on_train_epoch_start()

    with torch.no_grad():
        losses = transformer_encoder_module.training_step(batch, 0)

    for key in ("loss", "loss_MSE"):
        assert key in losses

    transformer_encoder_module.on_train_epoch_end()
    transformer_encoder_module.on_train_end()

    # validation
    dataloader = datamodule.val_dataloader()

    batch = next(iter(dataloader))

    transformer_encoder_module.on_validation_start()
    transformer_encoder_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = transformer_encoder_module.validation_step(batch, 0)

    transformer_encoder_module.on_validation_epoch_end()
    transformer_encoder_module.on_validation_end()

    for key in (
        "loss",
        "loss_MSE",
        "output",
    ):
        assert key in outputs
