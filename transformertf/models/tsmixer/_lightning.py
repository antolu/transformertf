from __future__ import annotations

import typing

import torch

from ...data import TransformerSample
from ...nn import QuantileLoss
from .._base_module import LR_CALL_TYPE, OPT_CALL_TYPE, LightningModuleBase
from ._config import TSMixerConfig
from ._model import TSMixer

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="TSMixerModule")


class TSMixerModule(LightningModuleBase):
    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        seq_len: int = 500,
        out_seq_len: int = 300,
        fc_dim: int = 1024,
        hidden_dim: int | None = None,
        num_blocks: int = 4,
        dropout: float = 0.1,
        activation: typing.Literal["relu", "gelu"] = "relu",
        norm: typing.Literal["batch", "layer"] = "batch",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        optimizer: str | OPT_CALL_TYPE = "adam",
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        reduce_on_plateau_patience: int = 200,
        max_epochs: int = 1000,
        validate_every_n_epochs: int = 50,
        log_grad_norm: bool = False,
        criterion: QuantileLoss | torch.nn.MSELoss | torch.nn.HuberLoss | None = None,
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
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self._lr_scheduler = lr_scheduler

        self.criterion = criterion or torch.nn.MSELoss()

        out_dim = 1
        if isinstance(self.criterion, QuantileLoss):
            out_dim = len(self.criterion.quantiles)

        self.model = TSMixer(
            num_feat=num_features,
            num_future_feat=num_features - 1,
            num_static_real_feat=num_static_features,
            seq_len=seq_len,
            out_seq_len=out_seq_len,
            fc_dim=fc_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_dim=out_dim,
        )

    @classmethod
    def parse_config_kwargs(
        cls, config: TSMixerConfig, **kwargs: typing.Any  # type: ignore[override]
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
                num_static_features=config.num_static_features,
                seq_len=config.ctxt_seq_len,
                out_seq_len=config.tgt_seq_len,
                dropout=config.dropout,
                activation=config.activation,
                fc_dim=config.fc_dim,
                hidden_dim=config.hidden_dim,
                num_blocks=config.num_blocks,
                norm=config.norm,
            )
        )

        default_kwargs.update(kwargs)

        # if num_features == 1:
        #     raise ValueError(
        #         "num_features must be greater than 1. "
        #         "Please specify input_columns in config, or "
        #         "pass in a different value for num_features."
        #     )

        return default_kwargs

    def forward(self, x: TransformerSample) -> torch.Tensor:
        return self.model(
            past_covariates=x["encoder_input"],
            future_covariates=x["decoder_input"][..., :-1],
        )

    def training_step(
        self, batch: TransformerSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        self.common_log_step({"loss": loss}, "train")

        return {"loss": loss}

    def validation_step(
        self, batch: TransformerSample, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))
        loss_dict = {"loss": loss}

        if not isinstance(self.criterion, torch.nn.MSELoss):
            loss_mse = torch.nn.MSELoss()(model_output, target)
            loss_dict["loss_MSE"] = loss_mse

        loss_mae = torch.nn.L1Loss()(model_output, target)
        loss_dict["loss_MAE"] = loss_mae

        self.common_log_step(loss_dict, "validation")

        return {"loss": loss, "output": model_output}
