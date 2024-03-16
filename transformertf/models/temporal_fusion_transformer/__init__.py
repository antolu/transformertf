from ._model import TemporalFusionTransformer  # noqa: F401
from ._config import TemporalFusionTransformerConfig  # noqa: F401

from ..._mod_replace import replace_modname


for _mod in (TemporalFusionTransformer, TemporalFusionTransformer):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

TFTConfig = TemporalFusionTransformerConfig

__all__ = [
    "TemporalFusionTransformer",
    "TemporalFusionTransformerConfig",
    "TFTConfig",
]
