from __future__ import annotations

import typing
from typing import NotRequired

import torch

from ...data import TimeSeriesSample
from ...nn import get_loss
from ...utils import ops
from .._base_module import LightningModuleBase
from ..typing import LR_CALL_TYPE, OPT_CALL_TYPE

if typing.TYPE_CHECKING:
    from ._config import LSTMConfig

    SameType = typing.TypeVar("SameType", bound="LSTMModule")

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    loss1: torch.Tensor
    loss2: torch.Tensor
    loss3: NotRequired[torch.Tensor]
    loss4: NotRequired[torch.Tensor]
    loss5: NotRequired[torch.Tensor]
    output: torch.Tensor
    state: HIDDEN_STATE


class LSTMModule(LightningModuleBase):
    def __init__(
        self,
        num_features: int = 1,
        num_layers: int = 3,
        sequence_length: int = 500,
        hidden_dim: int = 350,
        hidden_dim_fc: int = 1024,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        optimizer: str | OPT_CALL_TYPE = "ranger",
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        reduce_on_plateau_patience: int = 200,
        max_epochs: int = 1000,
        criterion: torch.nn.Module | None = None,
        lr_scheduler: str | LR_CALL_TYPE | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
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
        :param weight_decay: The optimizer weight decay, if applicable.
        :param momentum: The optimizer momentum, if applicable..
        :param optimizer: The optimizer to be used.
        :param optimizer_kwargs: Additional optimizer keyword arguments.
        :param reduce_on_plateau_patience: The number of epochs to wait before
            reducing the learning rate on a plateau.
        :param max_epochs: The maximum number of epochs to train for. This is
            used to determine the learning rate scheduler step size.
        :param phylstm: The PhyLSTM version to use. This may be 1, 2 or 3.
        :param criterion: The loss function to be used.
        :param lr_scheduler: The learning rate scheduler to be used.
        """
        criterion = criterion or torch.nn.MSELoss()
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
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
        )

        self._val_hidden: list[HIDDEN_STATE | None] = []  # type: ignore[assignment]

        self.model = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: HIDDEN_STATE | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: HIDDEN_STATE | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[torch.Tensor, HIDDEN_STATE]: ...

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: HIDDEN_STATE | None = None,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]:
        """
        Forward pass through the model.
        Rescales the output to the target scale if provided.

        :param x: The input sequence.
        :param hidden: The hidden states.
        :param target_scale: The target scale.
        :param return_states: Whether to return the hidden states.
        :return: The model output.
        """
        y_hat, hidden = self.model(x, hx=hidden_state)

        y_hat = self.fc(y_hat)

        if return_states:
            return y_hat, hidden
        return y_hat

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: LSTMConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        num_features = (
            len(config.input_columns) if config.input_columns is not None else 0
        )

        default_kwargs = {
            "num_features": num_features,
            "num_layers": config.num_layers,
            "sequence_length": config.seq_len,
            "hidden_dim": config.hidden_size,
            "hidden_dim_fc": config.hidden_size_fc,
            "dropout": config.dropout,
            "lr": config.lr,
            "max_epochs": config.num_epochs,
            "optimizer": config.optimizer,
            "optimizer_kwargs": config.optimizer_kwargs,
            "log_grad_norm": config.log_grad_norm,
            "criterion": get_loss(config.loss_fn),
            "lr_scheduler": config.lr_scheduler,
            "lr_scheduler_interval": config.lr_scheduler_interval,
        }

        default_kwargs.update(kwargs)

        return default_kwargs

    def on_validation_epoch_start(self) -> None:
        """Reset the hidden states"""
        self._val_hidden = [None]  # type: ignore[assignment]

        super().on_validation_epoch_start()

    def common_test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        hidden_state: HIDDEN_STATE | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, HIDDEN_STATE]:
        target = batch.get("target")
        assert target is not None

        hidden: HIDDEN_STATE
        output, hidden = self.forward(
            batch["input"],
            hidden_state=hidden_state,
            return_states=True,
        )

        hidden = ops.detach(hidden)  # type: ignore[assignment]

        loss = self.criterion(output, target)

        # also compute MSE and MAE
        loss1 = torch.nn.functional.mse_loss(output, target)
        loss2 = torch.nn.functional.l1_loss(output, target)

        loss_dict = {
            "loss": loss,
            "loss_MSE": loss1,
            "loss_MAE": loss2,
        }

        return loss_dict, output, hidden  # type: ignore[misc]

    def training_step(self, batch: TimeSeriesSample, batch_idx: int) -> torch.Tensor:
        target = batch.get("target")
        batch.get("target_scale")
        assert target is not None

        model_output = self.forward(batch["input"])

        loss = self.criterion(model_output, target)

        self.common_log_step({"loss": loss}, "train")

        return loss

    def validation_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        if dataloader_idx >= len(self._val_hidden):
            self._val_hidden.append(None)

        prev_hidden = self._val_hidden[dataloader_idx]

        loss, model_output, hidden = self.common_test_step(  # type: ignore[type-var]
            batch, batch_idx, prev_hidden
        )

        self._val_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        self.common_log_step(loss, "validation")

        return typing.cast(
            StepOutput,
            loss
            | {
                "state": ops.to_cpu(ops.detach(hidden)),
                "output": ops.to_cpu(ops.detach(model_output)),
            },
        )
