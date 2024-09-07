from .._mod_replace import replace_modname
from ._pete_predictor import PETEPredictor
from ._predictor_base import PredictorBase

for _mod in (PredictorBase, PETEPredictor):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
