from ..._mod_replace import replace_modname
from ._model import PhyTSMixer
from ._config import PhyTSMixerConfig
from ._lightning import PhyTSMixerModule

for _mod in (
    PhyTSMixer,
    PhyTSMixerConfig,
    PhyTSMixerModule,
):
    replace_modname(_mod, __name__)

del _mod


__all__ = [
    "PhyTSMixer",
    "PhyTSMixerConfig",
    "PhyTSMixerModule",
]
