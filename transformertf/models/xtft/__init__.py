from ..._mod_replace import replace_modname
from ._lightning import xTFT
from ._model import xTFTModel

for _mod in (
    xTFTModel,
    xTFT,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "xTFT",
    "xTFTModel",
]
