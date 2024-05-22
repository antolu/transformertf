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
