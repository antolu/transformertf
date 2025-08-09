from ..._mod_replace import replace_modname
from ._lightning import TransformerLSTM
from ._model import TransformerLSTMModel

replace_modname(TransformerLSTM, __name__)
replace_modname(TransformerLSTMModel, __name__)

del replace_modname

__all__ = [
    "TransformerLSTM",
    "TransformerLSTMModel",
]
