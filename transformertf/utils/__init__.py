from .._mod_replace import replace_modname
from . import _ops as ops
from . import _signal as signal
from ._compile import compile, maybe_compile, set_compile  # noqa: A004
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
    compile,
    set_compile,
    maybe_compile,
):
    replace_modname(func, __name__)


del replace_modname

__all__ = [
    "LrSchedulerDict",
    "OptimizerDict",
    "compile",
    "configure_lr_scheduler",
    "configure_optimizers",
    "maybe_compile",
    "ops",
    "set_compile",
    "signal",
]
