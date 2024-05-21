from __future__ import annotations

import pytest
import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.transformer_v2 import (
    TransformerV2Config,
    VanillaTransformerV2,
)

from ....conftest import CURRENT, DF_PATH


@pytest.fixture(scope="module")
def config() -> TransformerV2Config:
    return TransformerV2Config(
        fc_dim=64,
        n_dim_model=16,
        num_encoder_layers=2,
        num_decoder_layers=2,
        embedding="lstm",
        num_heads=2,
        train_dataset=[DF_PATH],
        val_dataset=[DF_PATH],
        batch_size=4,
        num_workers=0,
        target_depends_on=CURRENT,
        input_columns=["I_meas_A"],
        target_column="B_meas_T",
    )


@pytest.fixture(scope="module")
def transformer_v2_module(
    config: TransformerV2Config,
) -> VanillaTransformerV2:
    return VanillaTransformerV2.from_config(config)


@pytest.fixture()
def datamodule(config: TransformerV2Config) -> EncoderDecoderDataModule:
    return EncoderDecoderDataModule.from_parquet(config)


def test_transformer_v2_forward_pass_simple(
    transformer_v2_module: VanillaTransformerV2,
) -> None:
    x_past = torch.rand(
        1,
        TransformerV2Config.ctxt_seq_len,
        2,
    )
    x_future = torch.rand(
        1,
        TransformerV2Config.tgt_seq_len,
        2,
    )

    batch = {
        "encoder_input": x_past,
        "decoder_input": x_future,
    }

    with torch.no_grad():
        y = transformer_v2_module(batch)

    assert y.shape[:2] == x_future.shape[:2]


def test_transformer_v2_forward_pass_point(
    datamodule: EncoderDecoderDataModule,
    transformer_v2_module: VanillaTransformerV2,
) -> None:
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))

    transformer_v2_module.on_train_start()
    transformer_v2_module.on_train_epoch_start()

    with torch.no_grad():
        losses = transformer_v2_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    transformer_v2_module.on_train_epoch_end()
    transformer_v2_module.on_train_end()

    # validation
    dataloader = datamodule.val_dataloader()

    batch = next(iter(dataloader))

    transformer_v2_module.on_validation_start()
    transformer_v2_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = transformer_v2_module.validation_step(batch, 0)

    transformer_v2_module.on_validation_epoch_end()
    transformer_v2_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs


def test_transformer_v2_forward_pass_delta(
    datamodule: EncoderDecoderDataModule,
    transformer_v2_module: VanillaTransformerV2,
) -> None:
    transformer_v2_module.hparams["prediction_type"] = "delta"
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    batch = next(iter(dataloader))

    transformer_v2_module.on_train_start()
    transformer_v2_module.on_train_epoch_start()

    with torch.no_grad():
        losses = transformer_v2_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    transformer_v2_module.on_train_epoch_end()
    transformer_v2_module.on_train_end()

    # validation
    dataloader = datamodule.val_dataloader()

    batch = next(iter(dataloader))

    transformer_v2_module.on_validation_start()
    transformer_v2_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = transformer_v2_module.validation_step(batch, 0)

    transformer_v2_module.on_validation_epoch_end()
    transformer_v2_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs
