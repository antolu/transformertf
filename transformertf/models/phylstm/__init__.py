from ..._mod_replace import replace_modname
from ._datamodule import PhyLSTMDataModule
from ._lightning import PhyLSTM, StepOutput
from ._loss import LossWeights, PhyLSTMLoss
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
    LossWeights,
    PhyLSTMLoss,
    PhyLSTM1Model,
    PhyLSTM2Model,
    PhyLSTM3Model,
    PhyLSTM,
    PhyLSTMDataModule,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "LossWeights",
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
    "PhyLSTMDataModule",
    "PhyLSTMLoss",
    "StepOutput",
]
