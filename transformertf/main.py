from __future__ import annotations

import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401

from transformertf.data import (
    EncoderDataModule,  # noqa: F401
    EncoderDecoderDataModule,  # noqa: F401
    TimeSeriesDataModule,  # noqa: F401
)
from transformertf.models.lstm import LSTM  # noqa: F401
from transformertf.models.phylstm import PhyLSTMModule as PhyLSTM  # noqa: F401
from transformertf.models.temporal_fusion_transformer import (  # noqa: F401
    TemporalFusionTransformerModule as TemporalFusionTransformer,
)
from transformertf.models.transformer import (  # noqa: F401
    VanillaTransformerModule as VanillaTransformer,
)
from transformertf.models.transformer_v2 import (  # noqa: F401
    TransformerV2Module as VanillaTransformerV2,
)
from transformertf.models.tsmixer import TSMixerModule as TSMixer  # noqa: F401


def main() -> None:
    lightning.pytorch.cli.LightningCLI()
