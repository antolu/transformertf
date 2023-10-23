from .._mod_replace import replace_modname
from ._datamodule import DataModuleBase
from ._dataset import (
    TimeSeriesDataset,
    AbstractTimeSeriesDataset,
    TransformerDataset,
)
from ._sample_generator import (
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    TransformerSample,
    TransformerSampleGenerator,
)
from ._transform import (
    BaseTransform,
    PolynomialTransform,
    RunningNormalizer,
    TransformCollection,
)
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
    AbstractTimeSeriesDataset,
    TransformerDataset,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "AbstractTimeSeriesDataset",
    "BaseTransform",
    "DataModuleBase",
    "PolynomialTransform",
    "RunningNormalizer",
    "TimeSeriesDataset",
    "TimeSeriesSample",
    "TimeSeriesSampleGenerator",
    "TransformCollection",
    "TransformerDataset",
    "TransformerSample",
    "TransformerSampleGenerator",
    "WindowGenerator",
]
