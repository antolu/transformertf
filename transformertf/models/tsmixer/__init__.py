from ..._mod_replace import replace_modname
from ._lightning import TSMixer
from ._model import BasicTSMixerModel, TSMixerModel
from ._modules import (
    BatchNorm2D,
    ConditionalFeatureMixer,
    ConditionalMixerBlock,
    FeatureMixer,
    MixerBlock,
    TimeMixer,
)

for _mod in (
    MixerBlock,
    ConditionalMixerBlock,
    TimeMixer,
    FeatureMixer,
    ConditionalFeatureMixer,
    TSMixer,
    BatchNorm2D,
    BasicTSMixerModel,
    TSMixerModel,
):
    replace_modname(_mod, __name__)

del _mod


__all__ = [
    "BasicTSMixerModel",
    "BatchNorm2D",
    "ConditionalFeatureMixer",
    "ConditionalMixerBlock",
    "FeatureMixer",
    "MixerBlock",
    "TSMixer",
    "TSMixerModel",
    "TimeMixer",
]
