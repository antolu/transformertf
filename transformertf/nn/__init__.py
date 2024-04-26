from .._mod_replace import replace_modname
from ._add_norm import AddNorm
from ._get_activation import VALID_ACTIVATIONS, get_activation
from ._get_loss import VALID_LOSS, get_loss
from ._glu import GatedLinearUnit
from ._grn import GatedResidualNetwork
from ._interpretable_multi_head_attn import InterpretableMultiHeadAttention
from ._mlp import MLP
from ._quantile_loss import QuantileLoss
from ._resample_norm import ResampleNorm
from ._variable_selection import VariableSelection
from ._weighted_loss import WeightedHuberLoss, WeightedMAELoss, WeightedMSELoss

for _mod in (
    AddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    get_loss,
    get_activation,
    InterpretableMultiHeadAttention,
    QuantileLoss,
    ResampleNorm,
    VariableSelection,
    MLP,
    VALID_ACTIVATIONS,
    VALID_LOSS,
    WeightedMAELoss,
    WeightedMSELoss,
    WeightedHuberLoss,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "MLP",
    "VALID_ACTIVATIONS",
    "VALID_LOSS",
    "AddNorm",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "InterpretableMultiHeadAttention",
    "QuantileLoss",
    "ResampleNorm",
    "VariableSelection",
    "WeightedHuberLoss",
    "WeightedMAELoss",
    "WeightedMSELoss",
    "get_activation",
    "get_loss",
]
