from ..._mod_replace import replace_modname
from ._lightning import TemporalFusionTransformer
from ._model import TemporalFusionTransformerModel

for _mod in (
    TemporalFusionTransformerModel,
    TemporalFusionTransformerModel,
    TemporalFusionTransformer,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

TFT = TemporalFusionTransformerModel
TFTModule = TemporalFusionTransformer

__all__ = [
    "TFT",
    "TFTModule",
    "TemporalFusionTransformer",
    "TemporalFusionTransformerModel",
]
