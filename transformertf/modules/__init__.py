from .._mod_replace import replace_modname
from ._normalizer import RunningNormalizer

replace_modname(RunningNormalizer, __name__)

del replace_modname

__all__ = ["RunningNormalizer"]
