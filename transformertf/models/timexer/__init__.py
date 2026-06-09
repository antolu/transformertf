from __future__ import annotations

from ..._mod_replace import replace_modname
from ._model import TimeXerModel

for _mod in (TimeXerModel,):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = ["TimeXerModel"]
