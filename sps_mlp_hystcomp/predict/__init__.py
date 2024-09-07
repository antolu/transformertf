from .._mod_replace import replace_modname
from ._base_predictor import Predictor
from ._pete_predictor import PETEPredictor

for _mod in (Predictor, PETEPredictor):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
