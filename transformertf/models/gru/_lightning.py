from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import TimeSeriesSample
from ...nn import QuantileLoss
from ...utils import ops
from .._base_module import DEFAULT_LOGGING_METRICS, LightningModuleBase, MetricLiteral

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    state: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor | None]


class GRU(LightningModuleBase):
    def __init__(
        self,
        num_features: int = 1,
        num_layers: int = 3,
        n_dim_model: int = 350,
        output_dim: int = 1,
        dropout: float = 0.2,
        criterion: torch.nn.Module | None = None,
        *,
        compile_model: bool = False,
        log_grad_norm: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`torch.nn.LSTM` and defines
        the training, validation, testing and prediction steps.

        The model parameters are saved to the model directory by Lightning.

        Parameters
        ----------
        num_features: int
            The number of input features.
        num_layers: int
            The number of LSTM layers.
        n_dim_model: int
            The number of hidden units in each LSTM layer.
        n_dim_fc: int
            The number of hidden units in the fully connected layer.
        output_dim: int
            The number of output features. If the loss function is a quantile
            loss, this should be the number of quantiles.
        dropout: float
            The dropout probability.
        log_grad_norm: bool
            Whether to log the gradient norm.
        logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
            Container of metric names to compute and log during training, validation, and testing.
            If empty, no additional metrics will be logged (only the loss from the criterion).
            Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE".
        """
        super().__init__()

        self.criterion = criterion or torch.nn.MSELoss()
        self.save_hyperparameters(ignore=["criterion"])

        if isinstance(self.criterion, QuantileLoss):
            output_dim = self.criterion.num_quantiles
            self.hparams["output_dim"] = output_dim

        self._val_hidden: list[HIDDEN_STATE | None] = []

        self.model = torch.nn.GRU(
            input_size=num_features,
            hidden_size=n_dim_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(n_dim_model, output_dim)

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
            "hidden": ops.detach(hx),
            "output": ops.detach(model_output),
        }
        if isinstance(self.criterion, QuantileLoss):
            out["point_prediction"] = self.criterion.point_prediction(model_output)
        else:
            out["point_prediction"] = out["output"]

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
            "state": ops.detach(hidden),
            "output": ops.detach(model_output),
        }
        if isinstance(self.criterion, QuantileLoss):
            output_dict["point_prediction"] = self.criterion.point_prediction(
                model_output
            )
        else:
            output_dict["point_prediction"] = output_dict["output"]

        return typing.cast(StepOutput, loss | output_dict)
