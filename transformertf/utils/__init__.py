from .._mod_replace import replace_modname
from . import _ops as ops
from . import _signal as signal
from ._configure_optimizer import (
    LR_SCHEDULER_DICT,
    OPTIMIZER_DICT,
    configure_lr_scheduler,
    configure_optimizers,
)
from ._activation import get_activation

ops.__module__ = __name__ + "ops"
signal.__module__ = __name__ + "signal"

for func in (configure_lr_scheduler, configure_optimizers, get_activation):
    replace_modname(func, __name__)


del replace_modname

__all__ = [
    "ops",
    "signal",
    "LR_SCHEDULER_DICT",
    "OPTIMIZER_DICT",
    "configure_lr_scheduler",
    "configure_optimizers",
    "get_activation",
]
