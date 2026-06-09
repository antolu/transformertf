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
from ._model import TimeXerModel


class TimeXer(TransformerModuleBase):
    def __init__(
        self,
        num_past_covariates: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        patch_len: int = 16,
        d_ff: int = 512,
        dropout: float = 0.1,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        *,
        use_norm: bool = True,
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

        self.model = TimeXerModel(
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            num_past_covariates=num_past_covariates - 1,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_len=patch_len,
            d_ff=d_ff,
            dropout=dropout,
            use_norm=use_norm,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> dict[str, torch.Tensor]:
        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]

        x_enc = encoder_input[:, :, -1:]
        x_past_ex = encoder_input[:, :, :-1]
        x_ex = torch.cat([x_past_ex, decoder_input], dim=1)

        out = self.model(x_enc, x_ex)
        return {"output": out}
