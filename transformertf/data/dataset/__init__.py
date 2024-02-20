from ..._mod_replace import replace_modname
from ._dataset import (
    AbstractTimeSeriesDataset,
    EncoderDataset,
    EncoderDecoderDataset,
    TimeSeriesDataset,
)


for _mod in (
    TimeSeriesDataset,
    AbstractTimeSeriesDataset,
    EncoderDataset,
    EncoderDecoderDataset,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "AbstractTimeSeriesDataset",
    "EncoderDataset",
    "EncoderDecoderDataset",
    "TimeSeriesDataset",
]
