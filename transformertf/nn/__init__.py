from .._mod_replace import replace_modname
from ._add_norm import AddNorm
from ._glu import GatedLinearUnit
from ._grn import GatedResidualNetwork
from ._mlp import MLP
from ._quantile_loss import QuantileLoss

for _mod in (
    AddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    QuantileLoss,
    MLP,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "AddNorm",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "QuantileLoss",
    "MLP",
]
