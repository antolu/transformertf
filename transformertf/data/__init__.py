from transformertf.data.dataset import (
    AbstractTimeSeriesDataset,
    EncoderDecoderDataset,
    TimeSeriesDataset,
)

from .._mod_replace import replace_modname
from ._dataset_factory import DatasetFactory
from ._downsample import downsample
from ._sample_generator import (
    EncoderDecoderSample,
    EncoderDecoderTargetSample,
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    TransformerPredictionSampleGenerator,
    TransformerSampleGenerator,
)
from ._window_generator import WindowGenerator
from ._window_strategy import (
    TimeSeriesWindowStrategy,
    TransformerWindowStrategy,
    WindowStrategy,
    WindowStrategyFactory,
)
from .datamodule import (
    DataModuleBase,
    EncoderDecoderDataModule,
    TimeSeriesDataModule,
    TransformerDataModule,
)
from .transform import (
    AdaptiveSigmoidTransform,
    BaseTransform,
    DeltaTransform,
    DiscreteFunctionTransform,
    DivideByXTransform,
    FixedPolynomialTransform,
    Log1pTransform,
    LogTransform,
    MaxScaler,
    MinMaxScaler,
    PolynomialTransform,
    RunningNormalizer,
    SigmoidTransform,
    StandardScaler,
    TransformCollection,
)

for _mod in (
    WindowGenerator,
    TimeSeriesSample,
    EncoderDecoderSample,
    EncoderDecoderTargetSample,
    FixedPolynomialTransform,
    RunningNormalizer,
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
    TransformerPredictionSampleGenerator,
    DatasetFactory,
    TimeSeriesWindowStrategy,
    TransformerWindowStrategy,
    WindowStrategy,
    WindowStrategyFactory,
    downsample,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "AbstractTimeSeriesDataset",
    "AdaptiveSigmoidTransform",
    "BaseTransform",
    "DataModuleBase",
    "DatasetFactory",
    "DeltaTransform",
    "DiscreteFunctionTransform",
    "DivideByXTransform",
    "EncoderDecoderDataModule",
    "EncoderDecoderDataset",
    "EncoderDecoderSample",
    "EncoderDecoderTargetSample",
    "FixedPolynomialTransform",
    "Log1pTransform",
    "LogTransform",
    "MaxScaler",
    "MinMaxScaler",
    "PolynomialTransform",
    "RunningNormalizer",
    "SigmoidTransform",
    "StandardScaler",
    "TimeSeriesDataModule",
    "TimeSeriesDataset",
    "TimeSeriesSample",
    "TimeSeriesSampleGenerator",
    "TimeSeriesWindowStrategy",
    "TransformCollection",
    "TransformerDataModule",
    "TransformerPredictionSampleGenerator",
    "TransformerSampleGenerator",
    "TransformerWindowStrategy",
    "WindowGenerator",
    "WindowStrategy",
    "WindowStrategyFactory",
    "downsample",
]
