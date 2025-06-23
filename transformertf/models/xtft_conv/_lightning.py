from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_transformer import TransformerModuleBase
from ._model import xTFTConvModel


class xTFTConv(TransformerModuleBase):  # noqa: N801
    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        n_dim_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
        downsample_factor: int = 1,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        *,
        causal_attention: bool = True,
        prediction_type: typing.Literal["delta", "point"] = "point",
        log_grad_norm: bool = False,
        compile_model: bool = False,
        trainable_parameters: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        if criterion is None:
            criterion = QuantileLoss()
        if isinstance(criterion, QuantileLoss):
            self.hparams["output_dim"] = len(criterion.quantiles)
            output_dim = self.hparams["output_dim"]
        self.criterion = criterion

        self.model = xTFTConvModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            n_dim_model=n_dim_model,
            hidden_continuous_dim=hidden_continuous_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            output_dim=output_dim,
            downsample_factor=downsample_factor,
            causal_attention=causal_attention,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
        encoder_inputs = x["encoder_input"]
        decoder_inputs = x["decoder_input"][..., : self.hparams["num_future_features"]]

        return self.model(
            past_covariates=encoder_inputs,  # (B, T, F_past)
            future_covariates=decoder_inputs,
            encoder_lengths=x["encoder_lengths"][..., 0],  # (B, T)
            decoder_lengths=x["decoder_lengths"][..., 0],  # (B, T)
        )
