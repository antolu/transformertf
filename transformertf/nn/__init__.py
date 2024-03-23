from .._mod_replace import replace_modname
from ._add_norm import AddNorm
from ._glu import GatedLinearUnit
from ._grn import GatedResidualNetwork
from ._interpretable_multi_head_attn import InterpretableMultiHeadAttention
from ._mlp import MLP
from ._multi_embedding import MultiEmbedding
from ._quantile_loss import QuantileLoss
from ._resample_norm import ResampleNorm
from ._variable_selection import VariableSelection

for _mod in (
    AddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    MultiEmbedding,
    QuantileLoss,
    ResampleNorm,
    VariableSelection,
    MLP,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "AddNorm",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "InterpretableMultiHeadAttention",
    "MultiEmbedding",
    "QuantileLoss",
    "ResampleNorm",
    "VariableSelection",
    "MLP",
]
