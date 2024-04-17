from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_transformer import TransformerModuleBase
from ..typing import LR_CALL_TYPE, OPT_CALL_TYPE
from ._config import TemporalFusionTransformerConfig
from ._model import TemporalFusionTransformer

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="TemporalFusionTransformerModule")


class TemporalFusionTransformerModule(TransformerModuleBase):
    def __init__(
        self,
        num_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        n_dim_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        optimizer: str | OPT_CALL_TYPE = "adam",
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        reduce_on_plateau_patience: int = 200,
        max_epochs: int = 1000,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        prediction_type: typing.Literal["delta", "point"] = "point",
        lr_scheduler: str | LR_CALL_TYPE | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
        *,
        log_grad_norm: bool = False,
    ):
        if criterion is None:
            criterion = QuantileLoss()
            self.hparams["output_dim"] = len(criterion.quantiles)
            output_dim = self.hparams["output_dim"]

        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs or {},
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            max_epochs=max_epochs,
            log_grad_norm=log_grad_norm,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
            criterion=criterion,
        )

        self.model = TemporalFusionTransformer(
            num_features=num_features,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            n_dim_model=n_dim_model,
            num_static_features=1,
            hidden_continuous_dim=hidden_continuous_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            output_dim=output_dim,
        )

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: TemporalFusionTransformerConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        default_kwargs = super().parse_config_kwargs(config, **kwargs)
        num_features = (
            len(config.input_columns) if config.input_columns is not None else 0
        )
        num_features += 1  # add target

        default_kwargs |= {
            "num_features": num_features,
            "ctxt_seq_len": config.ctxt_seq_len,
            "tgt_seq_len": config.tgt_seq_len,
            "n_dim_model": config.n_dim_model,
            "hidden_continuous_dim": config.hidden_continuous_dim,
            "num_heads": config.num_heads,
            "num_lstm_layers": config.num_lstm_layers,
            "dropout": config.dropout,
            "output_dim": config.output_dim,
        }

        default_kwargs |= kwargs

        if num_features == 1:
            error_message = (
                "num_features must be greater than 1. "
                "Please specify input_columns in config, or "
                "pass in a different value for num_features."
            )
            raise ValueError(error_message)

        return default_kwargs

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
        return self.model(
            past_covariates=x["encoder_input"],
            future_covariates=x["decoder_input"],
            static_covariates=x["encoder_lengths"],  # type: ignore[typeddict-item]
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        model_output = self(batch)

        loss = self.calc_loss(model_output["output"], batch)

        loss_dict = {"loss": loss}
        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(point_prediction)

        self.common_log_step(loss_dict, "train")

        return {
            **loss_dict,
            "output": model_output["output"],
            "point_prediction": point_prediction,
        }

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        model_output = self(batch)

        loss = self.calc_loss(model_output["output"], batch)

        loss_dict = {"loss": loss}
        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(point_prediction)

        self.common_log_step(loss_dict, "validation")

        return {
            **loss_dict,
            "output": model_output["output"],
            "point_prediction": point_prediction,
        }
