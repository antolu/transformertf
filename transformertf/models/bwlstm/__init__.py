from ..._mod_replace import replace_modname
from ._lightning import BWLSTM, StepOutput
from ._loss import BoucWenLoss
from ._model import BWLSTM1, BWLSTM2, BWLSTM3
from ._output import (
    BoucWenOutput1,
    BoucWenOutput2,
    BoucWenOutput3,
    BoucWenStates1,
    BoucWenStates2,
    BoucWenStates3,
)

for _mod in (
    BoucWenLoss,
    BWLSTM1,
    BWLSTM2,
    BWLSTM3,
    BWLSTM,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "BWLSTM",
    "BWLSTM1",
    "BWLSTM2",
    "BWLSTM3",
    "BoucWenLoss",
    "BoucWenOutput1",
    "BoucWenOutput2",
    "BoucWenOutput3",
    "BoucWenStates1",
    "BoucWenStates2",
    "BoucWenStates3",
    "StepOutput",
]
