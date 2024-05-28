from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from .._base_module import LightningModuleBase
from ..bwlstm import BoucWenLoss, BoucWenOutput3
from ._model import PhyTSMixerModel


class PhyTSMixer(LightningModuleBase):
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
        criterion: BoucWenLoss | None = None,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.criterion = criterion or BoucWenLoss()

        self.model = PhyTSMixerModel(
            num_features=num_features,
            num_future_features=num_features - 1,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            fc_dim=fc_dim,
            hidden_dim=n_dim_model,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> BoucWenOutput3:
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
