from ..._mod_replace import replace_modname
from ._functional import mape_loss, smape_loss
from ._masked_mse import masked_mse_loss

for _mod in (mape_loss, smape_loss, masked_mse_loss):
    replace_modname(_mod, __name__)


del replace_modname
del _mod


__all__ = ["mape_loss", "masked_mse_loss", "smape_loss"]
