from ..._mod_replace import replace_modname
from ._lightning import xTFTConv
from ._model import xTFTConvModel

for _mod in (
    xTFTConvModel,
    xTFTConv,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "xTFTConv",
    "xTFTConvModel",
]
