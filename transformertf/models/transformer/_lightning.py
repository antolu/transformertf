from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_transformer import TransformerModuleBase
from ..typing import LR_CALL_TYPE, OPT_CALL_TYPE
from ._config import VanillaTransformerConfig
from ._model import VanillaTransformer

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="VanillaTransformerModule")


class VanillaTransformerModule(TransformerModuleBase):
    def __init__(
        self,
        num_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        n_dim_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        optimizer: str | OPT_CALL_TYPE = "adam",
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        reduce_on_plateau_patience: int = 200,
        max_epochs: int = 1000,
        criterion: QuantileLoss | None = None,
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

        self.model = VanillaTransformer(
            num_features=num_features,
            seq_len=ctxt_seq_len,
            out_seq_len=tgt_seq_len,
            n_dim_model=n_dim_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            fc_dim=fc_dim,
            output_dim=output_dim,
        )

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: VanillaTransformerConfig,
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
            "num_heads": config.num_heads,
            "num_encoder_layers": config.num_encoder_layers,
            "num_decoder_layers": config.num_decoder_layers,
            "dropout": config.dropout,
            "activation": config.activation,
            "fc_dim": config.fc_dim,
            "output_dim": config.output_dim,
        }

        default_kwargs |= kwargs

        if num_features == 1:
            msg = (
                "num_features must be greater than 1. "
                "Please specify input_columns in config, or "
                "pass in a different value for num_features."
            )
            raise ValueError(msg)

        return default_kwargs

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
        return self.model(
            source=x["encoder_input"],
            target=x["decoder_input"],
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        model_output = self(batch)

        loss = self.calc_loss(model_output, batch)

        loss_dict = {"loss": loss}
        point_prediction_dict: dict[str, torch.Tensor] = {}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}

        self.common_log_step(loss_dict, "train")

        return loss_dict | {"output": model_output} | point_prediction_dict

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        model_output = self(batch)

        loss = self.calc_loss(model_output, batch)

        loss_dict = {"loss": loss}
        point_prediction_dict: dict[str, torch.Tensor] = {}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}

        self.common_log_step(loss_dict, "validation")

        return loss_dict | {"output": model_output} | point_prediction_dict
