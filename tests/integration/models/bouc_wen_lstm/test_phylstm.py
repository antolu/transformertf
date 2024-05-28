"""
This module contains complete tests for the PhyLSTM model.
"""

from __future__ import annotations

import pandas as pd
import torch

from transformertf.data import TimeSeriesDataModule
from transformertf.models.bwlstm import (
    BWLSTM,
)

from ....conftest import FIELD


def test_phylstm_forward_pass(
    bouc_wen_module: BWLSTM,
    bouc_wen_datamodule: TimeSeriesDataModule,
    df: pd.DataFrame,
) -> None:
    bouc_wen_datamodule.prepare_data()
    bouc_wen_datamodule.setup()

    dataloader = bouc_wen_datamodule.train_dataloader()

    batch = next(iter(dataloader))

    bouc_wen_module.on_train_start()
    bouc_wen_module.on_train_epoch_start()

    with torch.no_grad():
        losses = bouc_wen_module.training_step(batch, 0)

    for key in ("loss", "loss1", "loss2", "loss3", "loss4", "loss5"):
        assert key in losses

    bouc_wen_module.on_train_epoch_end()
    bouc_wen_module.on_train_end()

    # validation
    dataloader = bouc_wen_datamodule.val_dataloader()

    batch = next(iter(dataloader))

    bouc_wen_module.on_validation_start()
    bouc_wen_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = bouc_wen_module.validation_step(batch, 0)

    bouc_wen_module.on_validation_epoch_end()
    bouc_wen_module.on_validation_end()

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
    dataloader = bouc_wen_datamodule.make_dataloader(df, predict=True)

    batch = next(iter(dataloader))

    bouc_wen_module.on_predict_start()
    bouc_wen_module.on_predict_epoch_start()

    with torch.no_grad():
        outputs_ = bouc_wen_module.predict_step(batch, 0)

    bouc_wen_module.on_predict_epoch_end()
    bouc_wen_module.on_predict_end()

    for key in ("output", "state"):
        assert key in outputs_
