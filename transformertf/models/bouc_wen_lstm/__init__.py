from ..._mod_replace import replace_modname
from ._lightning import BoucWenLSTM, StepOutput
from ._loss import BoucWenLoss
from ._model import BoucWenLSTMModel1, BoucWenLSTMModel2, BoucWenLSTMModel3
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
    BoucWenLSTMModel1,
    BoucWenLSTMModel2,
    BoucWenLSTMModel3,
    BoucWenLSTM,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "BoucWenLSTM",
    "BoucWenLSTMModel1",
    "BoucWenLSTMModel2",
    "BoucWenLSTMModel3",
    "BoucWenLoss",
    "BoucWenOutput1",
    "BoucWenOutput2",
    "BoucWenOutput3",
    "BoucWenStates1",
    "BoucWenStates2",
    "BoucWenStates3",
    "StepOutput",
]
