from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_transformer import TransformerModuleBase
from ._model import TransformerV2Model


class VanillaTransformerV2(TransformerModuleBase):
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
        embedding: typing.Literal["mlp", "lstm"] = "mlp",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        if criterion is None:
            criterion = QuantileLoss()
            self.hparams["output_dim"] = len(criterion.quantiles)
            output_dim = self.hparams["output_dim"]
        self.criterion = criterion

        self.model = TransformerV2Model(
            num_features=num_features,
            seq_len=ctxt_seq_len,
            out_seq_len=tgt_seq_len,
            n_dim_model=n_dim_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            embedding=embedding,
            fc_dim=fc_dim,
            output_dim=output_dim,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
        return self.model(
            source=x["encoder_input"],
            target=x["decoder_input"],
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
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
        model_output = self(batch)

        loss = self.calc_loss(model_output, batch)

        loss_dict = {"loss": loss}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}
        else:
            point_prediction_dict = {}

        self.common_log_step(loss_dict, "validation")

        return loss_dict | {"output": model_output} | point_prediction_dict
