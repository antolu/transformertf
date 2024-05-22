from __future__ import annotations

import typing

import torch

from ...data import EncoderTargetSample
from ...nn import VALID_ACTIVATIONS, QuantileLoss
from .._base_module import LightningModuleBase
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
        criterion: (QuantileLoss | torch.nn.MSELoss | torch.nn.HuberLoss | None) = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])
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
        self.criterion = criterion

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
