from ..._mod_replace import replace_modname
from ._lightning import PETE
from ._model import PETEModel

replace_modname(PETEModel, __name__)
replace_modname(PETE, __name__)

__all__ = ["PETE", "PETEModel"]
