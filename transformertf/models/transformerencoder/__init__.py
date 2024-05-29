from ..._mod_replace import replace_modname
from ._lightning import TransformerEncoderModule
from ._model import TransformerEncoder

for _mod in (
    TransformerEncoderModule,
    TransformerEncoder,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "TransformerEncoder",
    "TransformerEncoderModule",
]
