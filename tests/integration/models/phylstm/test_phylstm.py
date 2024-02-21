"""
This module contains complete tests for the PhyLSTM model.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from transformertf.models.phylstm import (
    PhyLSTMConfig,
    PhyLSTMDataModule,
    PhyLSTMModule,
)

from ....conftest import CURRENT, DF_PATH, FIELD


@pytest.fixture(scope="module")
def config() -> PhyLSTMConfig:
    config = PhyLSTMConfig(
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        num_workers=0,
        target_depends_on=CURRENT,
    )
    return config


@pytest.fixture
def phylstm_module(config: PhyLSTMConfig) -> PhyLSTMModule:
    return PhyLSTMModule.from_config(config)


@pytest.fixture
def phylstm_datamodule(config: PhyLSTMConfig) -> PhyLSTMDataModule:
    return PhyLSTMDataModule.from_parquet(config)


def test_phylstm_forward_pass(
    phylstm_module: PhyLSTMModule,
    phylstm_datamodule: PhyLSTMDataModule,
    df: pd.DataFrame,
) -> None:
    phylstm_datamodule.prepare_data()
    phylstm_datamodule.setup()

    dataloader = phylstm_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    phylstm_module.on_train_start()
    phylstm_module.on_train_epoch_start()

    with torch.no_grad():
        losses = phylstm_module.training_step(batch, 0)

    for key in ("loss", "loss1", "loss2", "loss3", "loss4", "loss5"):
        assert key in losses

    phylstm_module.on_train_epoch_end()
    phylstm_module.on_train_end()

    # validation
    dataloader = phylstm_datamodule.val_dataloader()

    batch = next(iter(dataloader))

    phylstm_module.on_validation_start()
    phylstm_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = phylstm_module.validation_step(batch, 0)

    phylstm_module.on_validation_epoch_end()
    phylstm_module.on_validation_end()

    for key in (
        "loss",
        "loss1",
        "loss2",
        "loss3",
        "loss4",
        "loss5",
        "output",
        "state",
    ):
        assert key in outputs

    # predict
    df = df.copy().drop(columns=[FIELD])
    dataloader = phylstm_datamodule.make_dataloader(df, predict=True)

    batch = next(iter(dataloader))

    phylstm_module.on_predict_start()
    phylstm_module.on_predict_epoch_start()

    with torch.no_grad():
        outputs_ = phylstm_module.predict_step(batch, 0)

    phylstm_module.on_predict_epoch_end()
    phylstm_module.on_predict_end()

    for key in ("output", "state"):
        assert key in outputs_
