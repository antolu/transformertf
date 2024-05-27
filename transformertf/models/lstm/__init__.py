from ..._mod_replace import replace_modname
from ._lightning import LSTM

for mod in [LSTM]:
    replace_modname(mod, __name__)

__all__ = ["LSTM"]
