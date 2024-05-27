from ..._mod_replace import replace_modname
from ._lightning import PhyTSMixer
from ._model import PhyTSMixerModel

for _mod in (
    PhyTSMixerModel,
    PhyTSMixer,
):
    replace_modname(_mod, __name__)

del _mod


__all__ = [
    "PhyTSMixer",
    "PhyTSMixerModel",
]
