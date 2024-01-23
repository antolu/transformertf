from .._mod_replace import replace_modname
from ._dataset import (
    AbstractTimeSeriesDataset,
    TimeSeriesDataset,
    TransformerDataset,
)
from ._sample_generator import (
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    TransformerSample,
    TransformerSampleGenerator,
)
from ._window_generator import WindowGenerator
from .datamodule import TimeSeriesDataModule, TransformerDataModule
from .transform import (
    BaseTransform,
    DiscreteFunctionTransform,
    FixedPolynomialTransform,
    PolynomialTransform,
    RunningNormalizer,
    TransformCollection,
)

for _mod in (
    WindowGenerator,
    TimeSeriesDataset,
    TimeSeriesSample,
    FixedPolynomialTransform,
    RunningNormalizer,
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
    "DiscreteFunctionTransform",
    "FixedPolynomialTransform",
    "PolynomialTransform",
    "RunningNormalizer",
    "TimeSeriesDataModule",
    "TimeSeriesDataset",
    "TimeSeriesSample",
    "TimeSeriesSampleGenerator",
    "TransformCollection",
    "TransformerDataModule",
    "TransformerDataset",
    "TransformerSample",
    "TransformerSampleGenerator",
    "WindowGenerator",
]
