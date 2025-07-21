from ..._mod_replace import replace_modname
from ._lightning import EncoderDecoderLSTM
from ._model import EncoderDecoderLSTMModel

for _mod in (
    EncoderDecoderLSTMModel,
    EncoderDecoderLSTM,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "EncoderDecoderLSTM",
    "EncoderDecoderLSTMModel",
]
