from ..._mod_replace import replace_modname
from ._lightning import VanillaTransformerV2
from ._model import TransformerV2Model

for _mod in (
    TransformerV2Model,
    VanillaTransformerV2,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "TransformerV2Model",
    "VanillaTransformerV2",
]
