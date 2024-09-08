from .._mod_replace import replace_modname
from ._base_predictor import Predictor
from ._pete_predictor import PETEPredictor
from ._tft_predictor import TFTPredictor

for _mod in (Predictor, PETEPredictor, TFTPredictor):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
