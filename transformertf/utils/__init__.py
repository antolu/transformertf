from .._mod_replace import replace_modname
from . import _ops as ops
from . import _signal as signal
from ._activation import VALID_ACTIVATIONS, get_activation
from ._configure_optimizer import (
    LrSchedulerDict,
    OptimizerDict,
    configure_lr_scheduler,
    configure_optimizers,
)
from .loss import VALID_LOSS, get_loss

ops.__module__ = __name__ + "ops"
signal.__module__ = __name__ + "signal"

for func in (
    configure_lr_scheduler,
    configure_optimizers,
    get_activation,
    VALID_ACTIVATIONS,
):
    replace_modname(func, __name__)


del replace_modname

__all__ = [
    "VALID_ACTIVATIONS",
    "VALID_LOSS",
    "LrSchedulerDict",
    "OptimizerDict",
    "configure_lr_scheduler",
    "configure_optimizers",
    "get_activation",
    "get_loss",
    "ops",
    "signal",
]
