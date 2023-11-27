from ..._mod_replace import replace_modname

from ._base import BaseTransform, TransformType, TransformCollection
from ._polynomial import PolynomialTransform, FixedPolynomialTransform
from ._scaler import RunningNormalizer
from ._discrete_fn import DiscreteFunctionTransform

StandardScaler = RunningNormalizer


for _mod in [
    BaseTransform,
    TransformType,
    TransformCollection,
    PolynomialTransform,
    FixedPolynomialTransform,
    RunningNormalizer,
    DiscreteFunctionTransform,
]:
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "BaseTransform",
    "TransformType",
    "TransformCollection",
    "PolynomialTransform",
    "FixedPolynomialTransform",
    "RunningNormalizer",
    "StandardScaler",
    "DiscreteFunctionTransform",
]
