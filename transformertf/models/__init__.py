from .._mod_replace import replace_modname
from ._base_module import LightningModuleBase
from ._base_transformer import TransformerModuleBase
from .attention_lstm import AttentionLSTM
from .bwlstm import BWLSTM1, BWLSTM2, BWLSTM3
from .encoder_decoder_lstm import EncoderDecoderLSTM
from .gru import GRU
from .lstm import LSTM
from .pete import PETE
from .pf_tft import PFTemporalFusionTransformer
from .sa_bwlstm import SABWLSTM
from .temporal_conv_transformer import TCT, TemporalConvTransformer
from .temporal_fusion_transformer import TemporalFusionTransformer
from .tft import TFT
from .transformer_v2 import VanillaTransformerV2
from .xtft import xTFT

for _mod in (
    LightningModuleBase,
    TransformerModuleBase,
    TemporalFusionTransformer,
    TFT,
    AttentionLSTM,
    EncoderDecoderLSTM,
    LSTM,
    GRU,
    SABWLSTM,
    PETE,
    PFTemporalFusionTransformer,
    xTFT,
    VanillaTransformerV2,
    BWLSTM1,
    BWLSTM2,
    BWLSTM3,
    TemporalConvTransformer,
    TCT,
):
    replace_modname(_mod, __name__)

del replace_modname

__all__ = [
    "BWLSTM1",
    "BWLSTM2",
    "BWLSTM3",
    "GRU",
    "LSTM",
    "PETE",
    "SABWLSTM",
    "TCT",
    "TFT",
    "AttentionLSTM",
    "EncoderDecoderLSTM",
    "LightningModuleBase",
    "PFTemporalFusionTransformer",
    "TemporalConvTransformer",
    "TemporalFusionTransformer",
    "TransformerModuleBase",
    "VanillaTransformerV2",
    "xTFT",
]
