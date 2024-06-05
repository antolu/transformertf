from .._mod_replace import replace_modname
from ._set_optimizer_params import SetOptimizerParamsCallback

replace_modname(SetOptimizerParamsCallback, __name__)


del replace_modname


__all__ = ["SetOptimizerParamsCallback"]
