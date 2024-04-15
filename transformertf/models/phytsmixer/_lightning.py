from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from .._base_module import LightningModuleBase
from ..phylstm import PhyLSTM3Output, PhyLSTMLoss
from ..typing import LR_CALL_TYPE, OPT_CALL_TYPE
from ._config import PhyTSMixerConfig
from ._model import PhyTSMixer

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="PhyTSMixerModule")


class PhyTSMixerModule(LightningModuleBase):
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
        criterion: PhyLSTMLoss | None = None,
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

        self.criterion = criterion or PhyLSTMLoss()

        self.model = PhyTSMixer(
            num_features=num_features,
            num_future_features=num_features - 1,
            input_len=seq_len,
            output_len=out_seq_len,
            fc_dim=fc_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: PhyTSMixerConfig,
        **kwargs: typing.Any,
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

        return default_kwargs

    def forward(self, x: EncoderDecoderTargetSample) -> PhyLSTM3Output:
        return self.model(
            past_covariates=x["encoder_input"],
            future_covariates=x["decoder_input"][..., :-1],
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        _, loss_dict = self.criterion(model_output, target, return_all=True)

        self.common_log_step(loss_dict, "train")

        return loss_dict | {"output": model_output}

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        _, loss_dict = self.criterion(model_output, target, return_all=True)

        self.common_log_step(loss_dict, "validation")

        return loss_dict | {"output": model_output}
