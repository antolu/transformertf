from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_module import LightningModuleBase
from ._model import TSMixerModel


class TSMixer(LightningModuleBase):
    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        ctxt_seq_len: int = 500,
        tgt_seq_len: int = 300,
        fc_dim: int = 1024,
        n_dim_model: int | None = None,
        num_blocks: int = 4,
        dropout: float = 0.1,
        activation: typing.Literal["relu", "gelu"] = "relu",
        norm: typing.Literal["batch", "layer"] = "batch",
        criterion: (QuantileLoss | torch.nn.MSELoss | torch.nn.HuberLoss | None) = None,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self.criterion = criterion or torch.nn.MSELoss()

        output_dim = 1
        if isinstance(self.criterion, QuantileLoss):
            output_dim = len(self.criterion.quantiles)

        self.hparams["output_dim"] = output_dim

        self.model = TSMixerModel(
            num_feat=num_features,
            num_future_feat=num_features - 1,
            num_static_real_feat=num_static_features,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            fc_dim=fc_dim,
            hidden_dim=n_dim_model,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
            output_dim=output_dim,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
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

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        self.common_log_step({"loss": loss}, "train")

        return {"loss": loss, "output": model_output}

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))
        loss_dict = {"loss": loss}

        point_prediction = model_output
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output).unsqueeze(
                -1
            )

        self.common_log_step(loss_dict, "validation")

        return {
            "loss": loss,
            "output": model_output,
            "point_prediction": point_prediction,
        }
