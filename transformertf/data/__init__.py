from transformertf.data.dataset import (
    AbstractTimeSeriesDataset,
    EncoderDecoderDataset,
    TimeSeriesDataset,
)

from .._mod_replace import replace_modname
from ._downsample import downsample
from ._sample_generator import (
    EncoderDecoderSample,
    EncoderDecoderTargetSample,
    EncoderTargetSample,
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    TransformerPredictionSampleGenerator,
    TransformerSampleGenerator,
)
from ._window_generator import WindowGenerator
from .datamodule import (
    DataModuleBase,
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
    TimeSeriesSample,
    EncoderTargetSample,
    EncoderDecoderSample,
    EncoderDecoderTargetSample,
    FixedPolynomialTransform,
    RunningNormalizer,
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
    TransformerPredictionSampleGenerator,
    downsample,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "AbstractTimeSeriesDataset",
    "BaseTransform",
    "DataModuleBase",
    "DiscreteFunctionTransform",
    "EncoderDecoderDataModule",
    "EncoderDecoderDataset",
    "EncoderDecoderSample",
    "EncoderDecoderTargetSample",
    "EncoderTargetSample",
    "FixedPolynomialTransform",
    "PolynomialTransform",
    "RunningNormalizer",
    "TimeSeriesDataModule",
    "TimeSeriesDataset",
    "TimeSeriesSample",
    "TimeSeriesSampleGenerator",
    "TransformCollection",
    "TransformerDataModule",
    "TransformerPredictionSampleGenerator",
    "TransformerSampleGenerator",
    "WindowGenerator",
    "downsample",
]
