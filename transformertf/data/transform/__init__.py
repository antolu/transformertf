from ..._mod_replace import replace_modname
from ._base import BaseTransform, TransformCollection
from ._delta import DeltaTransform
from ._discrete_fn import DiscreteFunctionTransform
from ._divide_by_x import DivideByXTransform
from ._log import Log1pTransform, LogTransform
from ._polynomial import FixedPolynomialTransform, PolynomialTransform
from ._scaler import MaxScaler, RunningNormalizer

StandardScaler = RunningNormalizer


for _mod in [
    BaseTransform,
    DeltaTransform,
    TransformCollection,
    LogTransform,
    Log1pTransform,
    MaxScaler,
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
    "DeltaTransform",
    "DiscreteFunctionTransform",
    "DivideByXTransform",
    "FixedPolynomialTransform",
    "Log1pTransform",
    "LogTransform",
    "MaxScaler",
    "PolynomialTransform",
    "RunningNormalizer",
    "StandardScaler",
    "TransformCollection",
]
