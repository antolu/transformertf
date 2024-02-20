from ..._mod_replace import replace_modname
from ._base import (
    AbstractTimeSeriesDataset,
)
from ._encoder import EncoderDataset
from ._encoder_decoder import EncoderDecoderDataset
from ._timeseries import TimeSeriesDataset


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
