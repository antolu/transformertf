from __future__ import annotations

from ..._mod_replace import replace_modname
from ._lightning import TimeXer
from ._model import TimeXerModel
from ._modules import (
    EncoderLayer,
    EnEmbedding,
    ExEmbedding,
    FlattenHead,
    PositionalEmbedding,
)

for _mod in (
    TimeXerModel,
    TimeXer,
    EnEmbedding,
    EncoderLayer,
    ExEmbedding,
    FlattenHead,
    PositionalEmbedding,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "EnEmbedding",
    "EncoderLayer",
    "ExEmbedding",
    "FlattenHead",
    "PositionalEmbedding",
    "TimeXer",
    "TimeXerModel",
]
