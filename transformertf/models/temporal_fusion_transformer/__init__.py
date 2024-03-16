from ._model import TemporalFusionTransformer  # noqa: F401

from ...._mod_replace import replace_modname


for _mod in (TemporalFusionTransformer,):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = ["TemporalFusionTransformer"]
