from .._mod_replace import replace_modname
from ._base_module import LightningModuleBase

for _mod in (LightningModuleBase,):
    replace_modname(_mod, __name__)

del replace_modname

__all__ = ["LightningModuleBase"]
