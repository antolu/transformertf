from ._checkpoint import CheckpointEvery
from .._mod_replace import replace_modname

replace_modname(CheckpointEvery, __name__)

__all__ = ["CheckpointEvery"]
