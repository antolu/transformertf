from .._mod_replace import replace_modname
from ._dataset import (
    AbstractTimeSeriesDataset,
    EncoderDataset,
    EncoderDecoderDataset,
    TimeSeriesDataset,
)
from ._downsample import downsample
from ._sample_generator import (
    EncoderDecoderTargetSample,
    EncoderTargetSample,
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
)
from ._window_generator import WindowGenerator
from .datamodule import (
    DataModuleBase,
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
    "DataModuleBase",
    "EncoderDataModule",
    "EncoderDecoderDataModule",
    "EncoderDataset",
    "EncoderDecoderDataset",
    "EncoderTargetSample",
    "EncoderDecoderTargetSample",
    "TransformerSampleGenerator",
    "WindowGenerator",
    "downsample",
]
