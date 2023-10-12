from ..._mod_replace import replace_modname
from ._config import PreisachConfig
from ._datamodule import PreisachDataModule
from ._lightning import PreisachModule

for mod in (PreisachConfig, PreisachDataModule, PreisachModule):
    replace_modname(mod, __name__)


__all__ = ["PreisachConfig", "PreisachDataModule", "PreisachModule"]
