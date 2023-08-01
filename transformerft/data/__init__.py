from .._mod_replace import replace_modname
from ._window_generator import WindowGenerator
from ._dataset import TimeSeriesDataset, TimeSeriesSample

for _mod in (WindowGenerator, TimeSeriesDataset, TimeSeriesSample):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = ["TimeSeriesDataset", "TimeSeriesSample", "WindowGenerator"]
