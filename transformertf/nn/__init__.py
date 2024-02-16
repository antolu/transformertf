from .._mod_replace import replace_modname
from ._add_and_norm import AddAndNorm
from ._glu import GatedLinearUnit
from ._grn import GatedResidualNetwork
from ._mlp import MLP
from ._quantile_loss import QuantileLoss

for _mod in (
    GatedLinearUnit,
    GatedResidualNetwork,
    AddAndNorm,
    QuantileLoss,
    MLP,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "AddAndNorm",
    "QuantileLoss",
    "MLP",
]
