from .._mod_replace import replace_modname
from ._datamodule import DataModuleBase
from ._dataset import TimeSeriesDataset, TimeSeriesSample
from ._transform import BaseTransform, PolynomialTransform, RunningNormalizer
from ._window_generator import WindowGenerator

for _mod in (
    DataModuleBase,
    WindowGenerator,
    TimeSeriesDataset,
    TimeSeriesSample,
    BaseTransform,
    PolynomialTransform,
    RunningNormalizer,
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
    "WindowGenerator",
]
