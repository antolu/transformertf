from .._mod_replace import replace_modname
from ._plot_hysteresis import PlotHysteresisCallback
from ._set_optimizer_lr import SetOptimizerLRCallback
from ._set_optimizer_params import SetOptimizerParamsCallback

replace_modname(SetOptimizerParamsCallback, __name__)
replace_modname(PlotHysteresisCallback, __name__)
replace_modname(SetOptimizerLRCallback, __name__)


del replace_modname


__all__ = [
    "PlotHysteresisCallback",
    "SetOptimizerLRCallback",
    "SetOptimizerParamsCallback",
]
