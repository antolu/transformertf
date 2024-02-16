from ..._mod_replace import replace_modname
from ._config import TransformerV2Config
from ._lightning import TransformerV2Module
from ._model import TransformerV2

for _mod in (
    TransformerV2,
    TransformerV2Config,
    TransformerV2Module,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "TransformerV2",
    "TransformerV2Config",
    "TransformerV2Module",
]
