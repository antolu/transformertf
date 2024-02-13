from .._mod_replace import replace_modname
from ._dataset import (
    AbstractTimeSeriesDataset,
    EncoderDataset,
    EncoderDecoderDataset,
    TimeSeriesDataset,
)
from ._downsample import downsample
from ._sample_generator import (
    EncoderDecoderSample,
    EncoderSample,
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
)
from ._window_generator import WindowGenerator
from .datamodule import (
    EncoderDataModule,
    EncoderDecoderDataModule,
    TimeSeriesDataModule,
    TransformerDataModule,
)
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
    "EncoderDataModule",
    "EncoderDecoderDataModule",
    "EncoderDataset",
    "EncoderDecoderDataset",
    "EncoderSample",
    "EncoderDecoderSample",
    "TransformerSampleGenerator",
    "WindowGenerator",
    "downsample",
]
