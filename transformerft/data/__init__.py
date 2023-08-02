from .._mod_replace import replace_modname
from ._datamodule import DataModuleBase
from ._dataset import TimeSeriesDataset, TimeSeriesSample
from ._window_generator import WindowGenerator

for _mod in (
    DataModuleBase,
    WindowGenerator,
    TimeSeriesDataset,
    TimeSeriesSample,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "DataModuleBase",
    "TimeSeriesDataset",
    "TimeSeriesSample",
    "WindowGenerator",
]
