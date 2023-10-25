from ..._mod_replace import replace_modname
from ._pos_enc import SimplePositionalEncoding
from ._model import VanillaTransformer
from ._config import VanillaTransformerConfig
from ._lightning import VanillaTransformerModule

for _mod in (
    SimplePositionalEncoding,
    VanillaTransformer,
    VanillaTransformerConfig,
    VanillaTransformerModule,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "SimplePositionalEncoding",
    "VanillaTransformer",
    "VanillaTransformerConfig",
    "VanillaTransformerModule",
]
