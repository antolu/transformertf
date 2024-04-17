from ..._mod_replace import replace_modname
from ._config import TransformerEncoderConfig
from ._lightning import TransformerEncoderModule
from ._model import TransformerEncoder

for _mod in (
    TransformerEncoderConfig,
    TransformerEncoderModule,
    TransformerEncoder,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "TransformerEncoder",
    "TransformerEncoderConfig",
    "TransformerEncoderModule",
]
