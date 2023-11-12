from ._base_module import LightningModuleBase
from .._mod_replace import replace_modname

for _mod in (LightningModuleBase,):
    replace_modname(_mod, __name__)

del replace_modname

__all__ = ["LightningModuleBase"]
