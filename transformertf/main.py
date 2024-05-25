from __future__ import annotations

import os
import typing

import jsonargparse
import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401
import torch
from lightning.pytorch.cli import LightningArgumentParser

from transformertf.data import (
    DataModuleBase,
    EncoderDataModule,  # noqa: F401
    EncoderDecoderDataModule,  # noqa: F401
    TimeSeriesDataModule,  # noqa: F401
)
from transformertf.models import LightningModuleBase
from transformertf.models.lstm import LSTM  # noqa: F401
from transformertf.models.phylstm import (  # noqa: F401
    PhyLSTM,
)
from transformertf.models.tsmixer import TSMixer  # noqa: F401


class LightningCLI(lightning.pytorch.cli.LightningCLI):
    model: torch.nn.Module

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, parser_kwargs={"parser_mode": "omegaconf"}, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.LearningRateMonitor, "lr_monitor"
        )
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.RichProgressBar, "progress_bar"
        )
        parser.set_defaults({"progress_bar.refresh_rate": 1})

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.RichModelSummary, "model_summary"
        )
        parser.set_defaults({"model_summary.max_depth": 2})

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.ModelCheckpoint, "checkpoint_every"
        )
        parser.set_defaults({
            "checkpoint_every.save_top_k": -1,
            "checkpoint_every.monitor": "loss/validation",
            "checkpoint_every.mode": "min",
            "checkpoint_every.dirpath": "checkpoints",
            "checkpoint_every.filename": "epoch={epoch}-valloss={loss/validation:.4f}",
            "checkpoint_every.every_n_epochs": 50,
        })

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.ModelCheckpoint, "checkpoint_best"
        )
        parser.set_defaults({
            "checkpoint_best.save_top_k": 1,
            "checkpoint_best.monitor": "loss/validation",
            "checkpoint_best.mode": "min",
            "checkpoint_best.dirpath": "checkpoints",
            "checkpoint_best.filename": "epoch={epoch}-valloss={loss/validation:.4f}",
            "checkpoint_best.save_last": "link",
            "checkpoint_best.save_weights_only": False,
            "checkpoint_best.auto_insert_metric_name": False,
            "checkpoint_best.enable_version_counter": False,
        })

        parser.set_defaults({
            "trainer.logger": jsonargparse.lazy_instance(
                lightning.pytorch.loggers.TensorBoardLogger,
                save_dir="logs",
                name=None,
            ),
        })

    def before_fit(self) -> None:
        # hijack model checkpoint callbacks to save to checkpoint_dir/version_{version}
        version = self.trainer.logger.version
        version_str = f"version_{version}"

        for callback in self.trainer.callbacks:
            if isinstance(callback, lightning.pytorch.callbacks.ModelCheckpoint):
                callback.dirpath = os.path.join(callback.dirpath, version_str)


def main() -> None:
    LightningCLI(
        subclass_mode_model=LightningModuleBase, subclass_mode_data=DataModuleBase
    )
