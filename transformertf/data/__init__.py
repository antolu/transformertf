from .._mod_replace import replace_modname
from ._dataset import (
    AbstractTimeSeriesDataset,
    TimeSeriesDataset,
    EncoderDataset,
    EncoderDecoderDataset,
)
from ._sample_generator import (
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    EncoderSample,
    EncoderDecoderSample,
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
from ._downsample import downsample

for _mod in (
    WindowGenerator,
    TimeSeriesDataset,
    TimeSeriesSample,
    FixedPolynomialTransform,
    RunningNormalizer,
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
    AbstractTimeSeriesDataset,
    EncoderDataset,
    EncoderDecoderDataset,
    downsample,
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
    "EncoderDataset",
    "EncoderDecoderDataset",
    "EncoderSample",
    "EncoderDecoderSample",
    "TransformerSampleGenerator",
    "WindowGenerator",
    "downsample",
]
