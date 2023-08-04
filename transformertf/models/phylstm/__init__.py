from ..._mod_replace import replace_modname
from ._config import PhyLSTMConfig
from ._datamodule import PhyLSTMDataModule
from ._lightning import STEP_OUTPUT, PhyLSTMModule
from ._loss import LossWeights, PhyLSTMLoss
from ._model import PhyLSTM1, PhyLSTM2, PhyLSTM3
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
    PhyLSTMConfig,
    PhyLSTM1,
    PhyLSTM2,
    PhyLSTM3,
    PhyLSTMModule,
    PhyLSTMDataModule,
):
    replace_modname(_mod, __name__)


del replace_modname
del _mod

__all__ = [
    "PhyLSTMLoss",
    "LossWeights",
    "PhyLSTMConfig",
    "PhyLSTM1",
    "PhyLSTM2",
    "PhyLSTM3",
    "PhyLSTMModule",
    "PhyLSTMDataModule",
    "STEP_OUTPUT",
    "PhyLSTM1Output",
    "PhyLSTM2Output",
    "PhyLSTM3Output",
    "PhyLSTM1States",
    "PhyLSTM2States",
    "PhyLSTM3States",
]
