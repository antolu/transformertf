from __future__ import annotations

import dataclasses

from ..phylstm import LossWeights
from ..tsmixer import TSMixerConfig


@dataclasses.dataclass
class PhyTSMixerConfig(TSMixerConfig):
    loss_weights: LossWeights | None = None
