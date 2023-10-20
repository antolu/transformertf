from .._mod_replace import replace_modname
from ._datamodule import DataModuleBase
from ._dataset import TimeSeriesDataset
from ._sample_generator import (TimeSeriesSample, TimeSeriesSampleGenerator,
                                TransformerSample, TransformerSampleGenerator)
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
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
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
    "TimeSeriesSampleGenerator",
    "TransformCollection",
    "TransformerSample",
    "TransformerSampleGenerator",
    "WindowGenerator",
]
