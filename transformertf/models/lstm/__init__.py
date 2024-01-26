from ._config import LSTMConfig
from ._lightning import LSTMModule

from ..._mod_replace import replace_modname

for mod in [LSTMConfig, LSTMModule]:
    replace_modname(mod, __name__)

__all__ = ["LSTMConfig", "LSTMModule"]
