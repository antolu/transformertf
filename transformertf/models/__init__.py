from .._mod_replace import replace_modname
from ._base_module import LightningModuleBase
from ._base_transformer import TransformerModuleBase

for _mod in (LightningModuleBase, TransformerModuleBase):
    replace_modname(_mod, __name__)

del replace_modname

__all__ = ["LightningModuleBase", "TransformerModuleBase"]
