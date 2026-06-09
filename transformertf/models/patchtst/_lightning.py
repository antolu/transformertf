from __future__ import annotations

import collections.abc

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_module import (
    DEFAULT_LOGGING_METRICS,
    MetricLiteral,
    setup_criterion_and_output_dim,
)
from .._base_transformer import TransformerModuleBase
from ._model import PatchTSTModel

__all__ = ["PatchTST"]


class PatchTST(TransformerModuleBase):
    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        patch_len: int = 16,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 256,
        d_lstm: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
        trainable_parameters: list[str] | None = None,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.criterion, _ = setup_criterion_and_output_dim(
            criterion,
            output_dim=1,
            default_quantiles=None,
        )

        self.model = PatchTSTModel(
            num_future_features=num_future_features,
            ctxt_seq_len=ctxt_seq_len,
            patch_len=patch_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            d_lstm=d_lstm,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            use_norm=use_norm,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> dict[str, torch.Tensor]:
        out = self.model(x["encoder_input"], x["decoder_input"])
        return {"output": out}
