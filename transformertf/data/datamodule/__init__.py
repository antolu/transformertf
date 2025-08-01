from ..._mod_replace import replace_modname
from ._base import DataModuleBase
from ._timeseries import TimeSeriesDataModule
from ._transformer import (
    EncoderDecoderDataModule,
    TransformerDataModule,
)

for _mod in (
    DataModuleBase,
    TransformerDataModule,
    TimeSeriesDataModule,
    EncoderDecoderDataModule,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "DataModuleBase",
    "EncoderDecoderDataModule",
    "TimeSeriesDataModule",
    "TransformerDataModule",
]
