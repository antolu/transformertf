from .._mod_replace import replace_modname
from ._log_hparams import LogHparamsCallback
from ._plot_bdot_hysteresis import PlotBdotHysteresisCallback
from ._plot_hysteresis import PlotHysteresisCallback
from ._set_optimizer_lr import SetOptimizerLRCallback
from ._set_optimizer_params import SetOptimizerParamsCallback

replace_modname(LogHparamsCallback, __name__)
replace_modname(SetOptimizerParamsCallback, __name__)
replace_modname(SetOptimizerLRCallback, __name__)
replace_modname(PlotHysteresisCallback, __name__)
replace_modname(PlotBdotHysteresisCallback, __name__)


del replace_modname


__all__ = [
    "LogHparamsCallback",
    "PlotBdotHysteresisCallback",
    "PlotHysteresisCallback",
    "SetOptimizerLRCallback",
    "SetOptimizerParamsCallback",
]
