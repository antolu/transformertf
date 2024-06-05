from ..._mod_replace import replace_modname
from ._lightning import BWLSTM1, BWLSTM2, BWLSTM3, StepOutput
from ._loss import BoucWenLoss
from ._model import BWLSTM1Model, BWLSTM2Model, BWLSTM3Model

for _mod in (
    BoucWenLoss,
    BWLSTM1Model,
    BWLSTM2Model,
    BWLSTM3Model,
    BWLSTM1,
    BWLSTM2,
    BWLSTM3,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "BWLSTM1",
    "BWLSTM2",
    "BWLSTM3",
    "BWLSTM1Model",
    "BWLSTM2Model",
    "BWLSTM3Model",
    "BoucWenLoss",
    "StepOutput",
]
