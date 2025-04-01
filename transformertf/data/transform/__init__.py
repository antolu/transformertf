"""
The :mod:`transformertf.data.transform` module includes classes for transforming data
in a scipy-like manner. The classes are designed to be used in a similar way to
scikit-learn transformers, which uses the `fit` and `transform` methods to learn
parameters from the data and apply the transformation, respectively. The classes
also include an `inverse_transform` method to reverse the transformation, and transforms
that fit parameters also export the parameters to a `state_dict`, to be saved with the data
processing pipeline in a serialized form.

The module includes the following classes:

- :class:`BaseTransform`: Base class for all transformers.
- :class:`DeltaTransform`: Calculates the difference between consecutive elements (similar to `np.diff`).
- :class:`DiscreteFunctionTransform`: Applies a discrete (piecewise linear) function to the input.
- :class:`DivideByXTransform`: Divides the target by the input.
- :class:`FixedPolynomialTransform`: Applies a fixed polynomial to the input.
- :class:`Log1pTransform`: Applies the natural logarithm to the input (+1).
- :class:`LogTransform`: Applies the natural logarithm to the input.
- :class:`MaxScaler`: Scales the input to the range [0, 1].
- :class:`PolynomialTransform`: Applies a polynomial to the input, fitted on the data.
- :class:`RunningNormalizer`: Scales the input to zero mean and unit variance. The :class:`StandardScaler` alias is also available.
- :class:`Sigmoid`: Applies the sigmoid function to the input.
- :class:`AdaptiveSigmoid`: Applies the sigmoid function to the input, with slope decreasing away from the center.

The module also includes the :class:`TransformCollection` class, which allows to chain
multiple transformers together, and apply them in sequence automatically.

Certain transforms like :class:`PolynomialTransform` and :class:`DiscreteFunctionTransform`
require two inputs (x, y) to fit the parameters. These transforms are of type `XY`,
whereas transforms like :class:`LogTransform` and :class:`MaxScaler` only require one input
and are of type `X`. Typically the user does not need to worry about this, as the
transformers will raise an error if the wrong number of inputs is provided, and a :class:`TransformCollection`
will automatically handle the input types.
"""

from ..._mod_replace import replace_modname
from ._base import BaseTransform, TransformCollection
from ._delta import DeltaTransform
from ._discrete_fn import DiscreteFunctionTransform
from ._divide_by_x import DivideByXTransform
from ._log import Log1pTransform, LogTransform
from ._polynomial import FixedPolynomialTransform, PolynomialTransform
from ._scaler import MaxScaler, RunningNormalizer
from ._sigmoid import AdaptiveSigmoidTransform, SigmoidTransform

StandardScaler = RunningNormalizer


for _mod in [
    BaseTransform,
    DeltaTransform,
    TransformCollection,
    LogTransform,
    Log1pTransform,
    MaxScaler,
    AdaptiveSigmoidTransform,
    SigmoidTransform,
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
    "AdaptiveSigmoidTransform",
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
    "SigmoidTransform",
    "StandardScaler",
    "TransformCollection",
]
