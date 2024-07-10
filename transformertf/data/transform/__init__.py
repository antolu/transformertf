from ..._mod_replace import replace_modname
from ._base import BaseTransform, TransformCollection, TransformType
from ._discrete_fn import DiscreteFunctionTransform
from ._divide_by_x import DivideByXTransform
from ._log import LogTransform
from ._polynomial import FixedPolynomialTransform, PolynomialTransform
from ._scaler import RunningNormalizer

StandardScaler = RunningNormalizer


for _mod in [
    BaseTransform,
    TransformType,
    TransformCollection,
    LogTransform,
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
    "LogTransform",
    "PolynomialTransform",
    "RunningNormalizer",
    "StandardScaler",
    "TransformCollection",
    "TransformType",
]
