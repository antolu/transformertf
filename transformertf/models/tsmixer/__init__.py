from ..._mod_replace import replace_modname

from ._lightning import TSMixerModule
from ._config import TSMixerConfig
from ._model import TSMixer

for _mod in [TSMixerModule, TSMixerConfig, TSMixer]:
    replace_modname(_mod, __name__)

del _mod


__all__ = ["TSMixerModule", "TSMixerConfig", "TSMixer"]
