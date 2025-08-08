from .._mod_replace import replace_modname
from ._add_norm import AddNorm
from ._gate_add_norm import GateAddNorm
from ._get_activation import VALID_ACTIVATIONS, get_activation
from ._get_loss import VALID_LOSS, get_loss
from ._glu import GatedLinearUnit
from ._grn import GatedResidualNetwork
from ._interpretable_multi_head_attn import InterpretableMultiHeadAttention
from ._mlp import MLP
from ._quantile_loss import QuantileLoss
from ._resample_norm import ResampleNorm
from ._temporal_conv_block import TemporalConvBlock
from ._temporal_decoder import TemporalDecoder
from ._temporal_encoder import TemporalEncoder
from ._transformer_encoder import (
    SimplePositionalEncoding,
    TransformerEncoder,
    generate_mask,
)
from ._variable_selection import VariableSelection
from ._weighted_loss import (
    HuberLoss,
    L1Loss,
    MAELoss,
    MAPELoss,
    MSELoss,
    SMAPELoss,
    WeightedHuberLoss,
    WeightedMAELoss,
    WeightedMSELoss,
)

for _mod in (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    get_loss,
    get_activation,
    InterpretableMultiHeadAttention,
    QuantileLoss,
    ResampleNorm,
    TemporalConvBlock,
    TemporalDecoder,
    TemporalEncoder,
    VariableSelection,
    MLP,
    VALID_ACTIVATIONS,
    VALID_LOSS,
    HuberLoss,
    L1Loss,
    MAELoss,
    MAPELoss,
    MSELoss,
    SMAPELoss,
    TransformerEncoder,
    SimplePositionalEncoding,
    generate_mask,
    WeightedHuberLoss,
    WeightedMAELoss,
    WeightedMSELoss,
):
    replace_modname(_mod, __name__)

del _mod
del replace_modname

__all__ = [
    "MLP",
    "VALID_ACTIVATIONS",
    "VALID_LOSS",
    "AddNorm",
    "GateAddNorm",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "HuberLoss",
    "InterpretableMultiHeadAttention",
    "L1Loss",
    "MAELoss",
    "MAPELoss",
    "MSELoss",
    "QuantileLoss",
    "ResampleNorm",
    "SMAPELoss",
    "SimplePositionalEncoding",
    "TemporalConvBlock",
    "TemporalDecoder",
    "TemporalEncoder",
    "TransformerEncoder",
    "VariableSelection",
    "WeightedHuberLoss",
    "WeightedMAELoss",
    "WeightedMSELoss",
    "generate_mask",
    "get_activation",
    "get_loss",
]
