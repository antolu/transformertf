from .._mod_replace import replace_modname
from ._datamodule import DataModuleBase
from ._dataset import TimeSeriesDataset, TimeSeriesSample
from ._transform import (BaseTransform, PolynomialTransform, RunningNormalizer,
                         TransformCollection)
from ._window_generator import WindowGenerator

for _mod in (
    DataModuleBase,
    WindowGenerator,
    TimeSeriesDataset,
    TimeSeriesSample,
    BaseTransform,
    PolynomialTransform,
    RunningNormalizer,
    TransformCollection,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "BaseTransform",
    "DataModuleBase",
    "PolynomialTransform",
    "RunningNormalizer",
    "TimeSeriesDataset",
    "TimeSeriesSample",
    "TransformCollection",
    "WindowGenerator",
]
