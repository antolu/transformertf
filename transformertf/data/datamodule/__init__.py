from ..._mod_replace import replace_modname
from ._base import _DataModuleBase
from ._timeseries import TimeSeriesDataModule
from ._transformer import (
    TransformerDataModule,
    EncoderDataModule,
    EncoderDecoderDataModule,
)

for _mod in (
    _DataModuleBase,
    TransformerDataModule,
    TimeSeriesDataModule,
    EncoderDataModule,
    EncoderDecoderDataModule,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = [
    "_DataModuleBase",
    "TransformerDataModule",
    "TimeSeriesDataModule",
    "EncoderDataModule",
    "EncoderDecoderDataModule",
]
