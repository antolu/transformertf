from __future__ import annotations

from ..._mod_replace import replace_modname
from ._lightning import TimeXer
from ._model import TimeXerModel

for _mod in (TimeXerModel, TimeXer):
    replace_modname(_mod, __name__)

del replace_modname
del _mod

__all__ = ["TimeXer", "TimeXerModel"]
