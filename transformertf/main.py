from __future__ import annotations

import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401

from transformertf.data import (
    EncoderDataModule,  # noqa: F401
    EncoderDecoderDataModule,  # noqa: F401
    TimeSeriesDataModule,  # noqa: F401
)
from transformertf.models.lstm import LSTM  # noqa: F401
from transformertf.models.phylstm import PhyLSTM  # noqa: F401
from transformertf.models.tsmixer import TSMixer  # noqa: F401


def main() -> None:
    lightning.pytorch.cli.LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})
