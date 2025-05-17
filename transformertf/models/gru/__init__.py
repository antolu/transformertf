from ..._mod_replace import replace_modname
from ._lightning import GRU

replace_modname(GRU, __name__)

del replace_modname

__all__ = ["GRU"]
