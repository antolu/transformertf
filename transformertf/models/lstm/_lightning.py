from __future__ import annotations

import typing

import torch

from ...data import TimeSeriesSample
from ...nn import QuantileLoss
from ...utils import ops
from .._base_module import LightningModuleBase

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    state: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor | None]


class LSTM(LightningModuleBase):
    def __init__(
        self,
        num_features: int = 1,
        num_layers: int = 3,
        seq_len: int = 500,
        hidden_dim: int = 350,
        hidden_dim_fc: int = 1024,
        output_dim: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        criterion: torch.nn.Module | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`torch.nn.LSTM` and defines
        the training, validation, testing and prediction steps.

        The model parameters are saved to the model directory by Lightning.

        :param num_layers: The number of LSTM layers.
        :param sequence_length: The length of the input sequence.
        :param hidden_dim: The number of hidden units in each LSTM layer.
        :param dropout: The dropout probability.
        :param lr: The optimizer learning rate. This may be set to "auto"
                   for use with the Lightning Learning Rate Finder.
        :param criterion: The loss function to be used.
        """
        super().__init__()

        self.criterion = criterion or torch.nn.MSELoss()
        self.save_hyperparameters(ignore=["criterion"])

        if isinstance(self.criterion, QuantileLoss):
            output_dim = self.criterion.num_quantiles
            self.hparams["output_dim"] = output_dim

        self._val_hidden: list[HIDDEN_STATE | None] = []

        self.model = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: HIDDEN_STATE | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: HIDDEN_STATE | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[torch.Tensor, HIDDEN_STATE]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: HIDDEN_STATE | None = None,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]:
        """
        Forward pass through the model.
        Rescales the output to the target scale if provided.

        :param x: The input sequence.
        :param hx: The hidden states.
        :param return_states: Whether to return the hidden states.
        :return: The model output.
        """
        y_hat, hidden = self.model(x, hx=hx)

        y_hat = self.fc(y_hat)

        if return_states:
            return y_hat, hidden
        return y_hat

    def on_validation_epoch_start(self) -> None:
        """Reset the hidden states"""
        self._val_hidden = [None]

        super().on_validation_epoch_start()

    def common_test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        hx: HIDDEN_STATE | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, HIDDEN_STATE]:
        target = batch.get("target")
        assert target is not None

        hidden: HIDDEN_STATE
        output, hidden = self.forward(
            batch["input"],
            hx=hx,
            return_states=True,
        )

        hidden = ops.detach(hidden)  # type: ignore[assignment]

        loss = self.criterion(output, target)

        loss_dict = {
            "loss": loss,
        }

        return loss_dict, output, hidden

    def training_step(self, batch: TimeSeriesSample, batch_idx: int) -> StepOutput:
        target = batch.get("target")
        assert target is not None

        model_output, hx = self.forward(batch["input"], return_states=True)
        loss = self.criterion(model_output, target)

        self.common_log_step({"loss": loss}, "train")

        out = {
            "loss": loss,
            "hidden": ops.to_cpu(ops.detach(hx)),
            "output": ops.to_cpu(ops.detach(model_output)),
        }
        if isinstance(self.criterion, QuantileLoss):
            out["point_prediction"] = self.criterion.point_prediction(model_output)

        return typing.cast(StepOutput, out)

    def validation_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        if dataloader_idx >= len(self._val_hidden):
            self._val_hidden.append(None)

        prev_hidden = self._val_hidden[dataloader_idx]

        loss, model_output, hidden = self.common_test_step(
            batch, batch_idx, prev_hidden
        )

        self._val_hidden[dataloader_idx] = hidden

        self.common_log_step(loss, "validation")

        output_dict = {
            "state": ops.to_cpu(ops.detach(hidden)),
            "output": ops.to_cpu(ops.detach(model_output)),
        }
        if isinstance(self.criterion, QuantileLoss):
            output_dict["point_prediction"] = self.criterion.point_prediction(
                model_output
            )

        return typing.cast(StepOutput, loss | output_dict)
