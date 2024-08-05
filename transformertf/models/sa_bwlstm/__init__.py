from ..._mod_replace import replace_modname
from ._lightning import SABWLSTM

replace_modname(SABWLSTM, __name__)

del replace_modname

__all__ = ["SABWLSTM"]
