from ..._mod_replace import replace_modname
from ._base import BaseTransform, TransformCollection
from ._discrete_fn import DiscreteFunctionTransform
from ._divide_by_x import DivideByXTransform
from ._log import Log1pTransform, LogTransform
from ._polynomial import FixedPolynomialTransform, PolynomialTransform
from ._scaler import RunningNormalizer

StandardScaler = RunningNormalizer


for _mod in [
    BaseTransform,
    TransformCollection,
    LogTransform,
    Log1pTransform,
    PolynomialTransform,
    FixedPolynomialTransform,
    RunningNormalizer,
    DiscreteFunctionTransform,
    DivideByXTransform,
]:
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "BaseTransform",
    "DiscreteFunctionTransform",
    "DivideByXTransform",
    "FixedPolynomialTransform",
    "Log1pTransform",
    "LogTransform",
    "PolynomialTransform",
    "RunningNormalizer",
    "StandardScaler",
    "TransformCollection",
]
