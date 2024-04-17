from .._mod_replace import replace_modname
from . import _ops as ops
from . import _signal as signal
from ._activation import ACTIVATIONS, get_activation
from ._configure_optimizer import (
    LrSchedulerDict,
    OptimizerDict,
    configure_lr_scheduler,
    configure_optimizers,
)

ops.__module__ = __name__ + "ops"
signal.__module__ = __name__ + "signal"

for func in (
    configure_lr_scheduler,
    configure_optimizers,
    get_activation,
    ACTIVATIONS,
):
    replace_modname(func, __name__)


del replace_modname

__all__ = [
    "ACTIVATIONS",
    "LrSchedulerDict",
    "OptimizerDict",
    "configure_lr_scheduler",
    "configure_optimizers",
    "get_activation",
    "ops",
    "signal",
]
