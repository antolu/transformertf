from __future__ import annotations

import typing

import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401
import torch
from lightning.pytorch.cli import LightningArgumentParser

from transformertf.data import (
    EncoderDataModule,  # noqa: F401
    EncoderDecoderDataModule,  # noqa: F401
    TimeSeriesDataModule,  # noqa: F401
)
from transformertf.models.lstm import LSTM  # noqa: F401
from transformertf.models.phylstm import (  # noqa: F401
    LossWeights,
    PhyLSTM,
    PhyLSTMLoss,
)
from transformertf.models.tsmixer import TSMixer  # noqa: F401


class LightningCLI(lightning.pytorch.cli.LightningCLI):
    model: torch.nn.Module

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, parser_kwargs={"parser_mode": "omegaconf"}, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--no-compile",
            action="store_true",
            dest="no_compile",
            help="Do not compile the model with torch.",
        )

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.LearningRateMonitor, "lr_monitor"
        )
        parser.set_defaults({"lr_monitor": {"logging_interval": "epoch"}})

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.RichProgressBar, "progress_bar"
        )
        parser.set_defaults({"progress_bar": {"refresh_rate": 1}})

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.RichModelSummary, "model_summary"
        )
        parser.set_defaults({"model_summary": {"max_depth": 2}})

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.ModelCheckpoint, "checkpoint_every"
        )
        parser.set_defaults({
            "checkpoint_every": {
                "save_top_k": -1,
                "monitor": "loss/validation",
                "mode": "min",
                "every_n_epochs": 1,
                "filename": "epoch={epoch}-every-valloss={loss/validation:.4f}",
            }
        })

        parser.add_lightning_class_args(
            lightning.pytorch.callbacks.ModelCheckpoint, "checkpoint_best"
        )
        parser.set_defaults({
            "checkpoint_best": {
                "save_top_k": 1,
                "monitor": "loss/validation",
                "mode": "min",
                "dirpath": "checkpoints",
                "filename": "epoch={epoch}-valloss={loss/validation:.4f}",
                "save_last": "link",
                "save_weights_only": False,
                "auto_insert_metric_name": False,
                "enable_version_counter": False,
            }
        })

        parser.set_defaults({
            "trainer": {
                "logger": {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {
                        "log_graph": True,
                        "save_dir": "logs",
                        "name": None,
                    },
                }
            }
        })

    def before_fit(self) -> None:
        self.model = self._maybe_compile(self.model)

    def before_validate(self) -> None:
        self.model = self._maybe_compile(self.model)

    def before_test(self) -> None:
        self.model = self._maybe_compile(self.model)

    def before_predict(self) -> None:
        self.model = self._maybe_compile(self.model)

    def _maybe_compile(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.config.fit.no_compile and not isinstance(
            self.model,
            torch._dynamo.eval_frame.OptimizedModule,  # noqa: SLF001
        ):
            model = torch.compile(model)
        return model


def main() -> None:
    LightningCLI()
