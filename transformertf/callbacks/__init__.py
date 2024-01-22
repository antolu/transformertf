from .._mod_replace import replace_modname
from ._checkpoint import CheckpointEvery

replace_modname(CheckpointEvery, __name__)

__all__ = ["CheckpointEvery"]
