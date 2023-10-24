from ..._mod_replace import replace_modname
from ._pos_enc import SimplePositionalEncoding
from ._model import VanillaTransformer

for _mod in (SimplePositionalEncoding, VanillaTransformer):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = ["SimplePositionalEncoding", "VanillaTransformer"]
