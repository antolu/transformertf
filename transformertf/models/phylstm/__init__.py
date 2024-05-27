from ..._mod_replace import replace_modname
from ._lightning import PhyLSTM, StepOutput
from ._loss import PhyLSTMLoss
from ._model import PhyLSTM1Model, PhyLSTM2Model, PhyLSTM3Model
from ._output import (
    PhyLSTM1Output,
    PhyLSTM1States,
    PhyLSTM2Output,
    PhyLSTM2States,
    PhyLSTM3Output,
    PhyLSTM3States,
)

for _mod in (
    PhyLSTMLoss,
    PhyLSTM1Model,
    PhyLSTM2Model,
    PhyLSTM3Model,
    PhyLSTM,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "PhyLSTM",
    "PhyLSTM1Model",
    "PhyLSTM1Output",
    "PhyLSTM1States",
    "PhyLSTM2Model",
    "PhyLSTM2Output",
    "PhyLSTM2States",
    "PhyLSTM3Model",
    "PhyLSTM3Output",
    "PhyLSTM3States",
    "PhyLSTMLoss",
    "StepOutput",
]
