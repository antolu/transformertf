from .._mod_replace import replace_modname
from ._window_generator import WindowGenerator

replace_modname(WindowGenerator, __name__)

del replace_modname

__all__ = ["WindowGenerator"]
