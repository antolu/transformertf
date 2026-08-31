from __future__ import annotations

from ..._mod_replace import replace_modname
from ._lightning import PatchTST
from ._model import PatchTSTModel
from ._modules import (
    BahdanauAttention,
    LSTMDecoderWithAttention,
    PatchEmbedding,
    SpatialAttentionBlock,
    STEncoderBlock,
    TemporalAttentionBlock,
)

for _mod in (
    PatchTST,
    PatchTSTModel,
    BahdanauAttention,
    LSTMDecoderWithAttention,
    PatchEmbedding,
    SpatialAttentionBlock,
    STEncoderBlock,
    TemporalAttentionBlock,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "BahdanauAttention",
    "LSTMDecoderWithAttention",
    "PatchEmbedding",
    "PatchTST",
    "PatchTSTModel",
    "STEncoderBlock",
    "SpatialAttentionBlock",
    "TemporalAttentionBlock",
]
