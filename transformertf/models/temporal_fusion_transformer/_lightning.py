from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from ...utils.loss import get_loss
from .._base_module import LightningModuleBase
from ..typing import LR_CALL_TYPE, OPT_CALL_TYPE
from ._config import TemporalFusionTransformerConfig
from ._model import TemporalFusionTransformer

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar(
        "SameType", bound="TemporalFusionTransformerModule"
    )


class TemporalFusionTransformerModule(LightningModuleBase):
    def __init__(
        self,
        num_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        n_dim_model: int = 300,
        variable_selection_dim: int = 100,
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
        validate_every_n_epochs: int = 50,
        log_grad_norm: bool = False,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        prediction_type: typing.Literal["delta", "point"] = "point",
        lr_scheduler: str | LR_CALL_TYPE | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs or {},
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            max_epochs=max_epochs,
            validate_every_n_epochs=validate_every_n_epochs,
            log_grad_norm=log_grad_norm,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
        )
        if criterion is None:
            self.criterion = QuantileLoss()
            output_dim = 7
        else:
            self.criterion = criterion
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self.model = TemporalFusionTransformer(
            num_features=num_features,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            n_dim_model=n_dim_model,
            variable_selection_dim=variable_selection_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            output_dim=output_dim,
        )

    @classmethod
    def parse_config_kwargs(
        cls, config: TemporalFusionTransformerConfig, **kwargs: typing.Any  # type: ignore[override]
    ) -> dict[str, typing.Any]:
        default_kwargs = super().parse_config_kwargs(config, **kwargs)
        num_features = (
            len(config.input_columns)
            if config.input_columns is not None
            else 0
        )
        num_features += 1  # add target

        default_kwargs.update(
            dict(
                num_features=num_features,
                ctxt_seq_len=config.ctxt_seq_len,
                tgt_seq_len=config.tgt_seq_len,
                n_dim_model=config.n_dim_model,
                variable_selection_dim=config.variable_selection_dim,
                num_heads=config.num_heads,
                num_lstm_layers=config.num_lstm_layers,
                dropout=config.dropout,
                output_dim=config.output_dim,
                prediction_type=config.prediction_type,
            )
        )

        default_kwargs.update(kwargs)

        if "loss_fn" in default_kwargs and isinstance(
            default_kwargs["loss_fn"], str
        ):
            default_kwargs["criterion"] = get_loss(default_kwargs["loss_fn"])  # type: ignore[arg-type]

        if num_features == 1:
            raise ValueError(
                "num_features must be greater than 1. "
                "Please specify input_columns in config, or "
                "pass in a different value for num_features."
            )

        return default_kwargs

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
        return self.model(
            past_covariates=x["encoder_input"],
            future_covariates=x["decoder_input"],
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"].squeeze(-1)

        model_output = self(batch)

        loss = self.criterion(model_output["output"], target)

        loss_dict = {"loss": loss}
        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(
                point_prediction
            )

        if self.hparams["prediction_type"] == "point":
            loss_dict["loss_MSE"] = torch.nn.functional.mse_loss(
                point_prediction, target
            )

        self.common_log_step(loss_dict, "train")

        return loss_dict

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"].squeeze(-1)

        model_output = self(batch)

        loss = self.criterion(model_output["output"], target)

        loss_dict = {"loss": loss}
        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(
                point_prediction
            )

        if self.hparams["prediction_type"] == "point":
            loss_dict["loss_MSE"] = torch.nn.functional.mse_loss(
                point_prediction, target
            )

        self.common_log_step(loss_dict, "validation")

        return {**loss_dict, "output": model_output}
