from ..._mod_replace import replace_modname
from ._config import LSTMConfig
from ._lightning import LSTMModule

for mod in [LSTMConfig, LSTMModule]:
    replace_modname(mod, __name__)

__all__ = ["LSTMConfig", "LSTMModule"]
