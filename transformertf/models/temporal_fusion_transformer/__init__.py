from ..._mod_replace import replace_modname
from ._config import TemporalFusionTransformerConfig
from ._lightning import TemporalFusionTransformerModule
from ._model import TemporalFusionTransformer

for _mod in (
    TemporalFusionTransformer,
    TemporalFusionTransformer,
    TemporalFusionTransformerModule,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

TFT = TemporalFusionTransformer
TFTConfig = TemporalFusionTransformerConfig
TFTModule = TemporalFusionTransformerModule

__all__ = [
    "TFT",
    "TFTConfig",
    "TFTModule",
    "TemporalFusionTransformer",
    "TemporalFusionTransformerConfig",
    "TemporalFusionTransformerModule",
]
