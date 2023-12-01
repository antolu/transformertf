from .._mod_replace import replace_modname

from ._quantile_loss import QuantileLoss

replace_modname(QuantileLoss, __name__)
del replace_modname

__all__ = ["QuantileLoss"]
