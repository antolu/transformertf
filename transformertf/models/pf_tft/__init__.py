from ..._mod_replace import replace_modname
from ._lightning import PFTemporalFusionTransformer
from ._model import PFTemporalFusionTransformerModel

for _mod in (
    PFTemporalFusionTransformerModel,
    PFTemporalFusionTransformer,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "PFTemporalFusionTransformer",
    "PFTemporalFusionTransformerModel",
]
