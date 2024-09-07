from .._mod_replace import replace_modname
from ._base_predictor import BasePredictor
from ._pete_predictor import PETEPredictor

for _mod in (BasePredictor, PETEPredictor):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
