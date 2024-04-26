from __future__ import annotations

import typing

import torch

from ...data import EncoderTargetSample
from ...nn import VALID_ACTIVATIONS, QuantileLoss, get_loss
from .._base_module import LightningModuleBase
from ..typing import LR_CALL_TYPE, OPT_CALL_TYPE
from ._config import TransformerEncoderConfig
from ._model import TransformerEncoder

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="TransformerEncoderModule")


class TransformerEncoderModule(LightningModuleBase):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        n_dim_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        optimizer: str | OPT_CALL_TYPE = "adam",
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        reduce_on_plateau_patience: int = 200,
        max_epochs: int = 1000,
        criterion: (QuantileLoss | torch.nn.MSELoss | torch.nn.HuberLoss | None) = None,
        lr_scheduler: str | LR_CALL_TYPE | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
        *,
        log_grad_norm: bool = False,
    ):
        if isinstance(criterion, QuantileLoss):
            if output_dim != len(criterion.quantiles):
                msg = (
                    "output_dim must be equal to the number of quantiles "
                    "in the QuantileLoss criterion."
                )
                raise ValueError(msg)
        elif criterion is not None and output_dim != 1:
            msg = "output_dim must be 1 if a custom criterion is specified."
            raise ValueError(msg)
        elif criterion is None and output_dim != 7:
            msg = (
                "output_dim must be 7 if criterion is None as "
                "default criterion is QuantileLoss. "
                "Otherwise, specify a custom criterion."
            )
            raise ValueError(msg)
        elif criterion is None:
            criterion = QuantileLoss()

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
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self.model = TransformerEncoder(
            num_features=num_features,
            seq_len=seq_len,
            n_dim_model=n_dim_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation,
            fc_dim=fc_dim,
            output_dim=output_dim,
        )

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: TransformerEncoderConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        default_kwargs = super().parse_config_kwargs(config, **kwargs)
        num_features = (
            len(config.input_columns) if config.input_columns is not None else 0
        )
        num_features += 1  # add target

        if num_features == 1:
            msg = (
                "num_features must be greater than 1. "
                "Please specify input_columns in config, or "
                "pass in a different value for num_features."
            )
            raise ValueError(msg)

        default_kwargs.update({
            "num_features": num_features,
            "seq_len": config.ctxt_seq_len + config.tgt_seq_len,
            "n_dim_model": config.n_dim_model,
            "num_heads": config.num_heads,
            "num_encoder_layers": config.num_encoder_layers,
            "dropout": config.dropout,
            "activation": config.activation,
            "fc_dim": config.fc_dim,
            "output_dim": config.output_dim,
        })

        default_kwargs.update(kwargs)

        if "criterion" in default_kwargs and isinstance(
            default_kwargs["criterion"], str
        ):
            default_kwargs["criterion"] = get_loss(default_kwargs["criterion"])  # type: ignore[arg-type]

        return default_kwargs

    def forward(self, x: EncoderTargetSample) -> torch.Tensor:
        return self.model(
            source=x["encoder_input"],
        )

    def training_step(
        self, batch: EncoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"].squeeze(-1)

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        loss_dict = {"loss": loss}
        point_prediction_dict: dict[str, torch.Tensor] = {}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}

        self.common_log_step(loss_dict, "train")

        return loss_dict | {"output": model_output} | point_prediction_dict

    def validation_step(
        self,
        batch: EncoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"].squeeze(-1)

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        loss_dict = {"loss": loss}
        point_prediction_dict: dict[str, torch.Tensor] = {}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}

        self.common_log_step(loss_dict, "validation")

        return loss_dict | {"output": model_output} | point_prediction_dict
