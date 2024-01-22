from ..._mod_replace import replace_modname
from ._config import TSMixerConfig
from ._lightning import TSMixerModule
from ._model import BasicTSMixer, TSMixer
from ._modules import (
    TimeMixer,
    FeatureMixer,
    ConditionalFeatureMixer,
    MixerBlock,
    ConditionalMixerBlock,
    BatchNorm2D,
)

for _mod in (
    MixerBlock,
    ConditionalMixerBlock,
    TimeMixer,
    FeatureMixer,
    ConditionalFeatureMixer,
    TSMixerModule,
    TSMixerConfig,
    BatchNorm2D,
    BasicTSMixer,
    TSMixer,
):
    replace_modname(_mod, __name__)

del _mod


__all__ = [
    "MixerBlock",
    "TimeMixer",
    "FeatureMixer",
    "ConditionalFeatureMixer",
    "ConditionalMixerBlock",
    "BatchNorm2D",
    "TSMixerModule",
    "TSMixerConfig",
    "BasicTSMixer",
    "TSMixer",
]
