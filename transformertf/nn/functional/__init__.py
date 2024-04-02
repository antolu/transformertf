from ._functional import mape_loss, smape_loss

from ..._mod_replace import replace_modname


for _mod in (mape_loss, smape_loss):
    replace_modname(_mod, __name__)


del replace_modname
del _mod


__all__ = ["mape_loss", "smape_loss"]
