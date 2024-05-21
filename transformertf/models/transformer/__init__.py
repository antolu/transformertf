from ..._mod_replace import replace_modname
from ._lightning import VanillaTransformer
from ._model import VanillaTransformerModel
from ._pos_enc import SimplePositionalEncoding

for _mod in (
    SimplePositionalEncoding,
    VanillaTransformerModel,
    VanillaTransformer,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "SimplePositionalEncoding",
    "VanillaTransformer",
    "VanillaTransformerModel",
]
