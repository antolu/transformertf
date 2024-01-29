from __future__ import annotations

import dataclasses

from ..tsmixer import TSMixerConfig
from ..phylstm import LossWeights


@dataclasses.dataclass
class PhyTSMixerConfig(TSMixerConfig):
    loss_weights: LossWeights | None = None
