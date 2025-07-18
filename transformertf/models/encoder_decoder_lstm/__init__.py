from ..._mod_replace import replace_modname
from ._lightning import EncoderDecoderLSTMModule
from ._model import EncoderDecoderLSTM

for _mod in (
    EncoderDecoderLSTM,
    EncoderDecoderLSTMModule,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "EncoderDecoderLSTM",
    "EncoderDecoderLSTMModule",
]
