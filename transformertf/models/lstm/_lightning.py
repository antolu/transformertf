from __future__ import annotations

import sys
import typing

if sys.version_info >= (3, 10):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

import torch

from ...data import TimeSeriesSample
from ...utils import ops
from .._base_module import LR_CALL_TYPE, OPT_CALL_TYPE, LightningModuleBase

if typing.TYPE_CHECKING:
    from ._config import LSTMConfig

    SameType = typing.TypeVar("SameType", bound="LSTMModule")

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]

STEP_OUTPUT = typing.TypedDict(
    "STEP_OUTPUT",
    {
        "loss": torch.Tensor,
        "loss1": torch.Tensor,
        "loss2": torch.Tensor,
        "loss3": NotRequired[torch.Tensor],
        "loss4": NotRequired[torch.Tensor],
        "loss5": NotRequired[torch.Tensor],
        "output": torch.Tensor,
        "state": HIDDEN_STATE,
    },
)


LOSS_FN = typing.Literal["mse", "huber", "l1"]
CRITERION: dict[LOSS_FN, torch.nn.Module] = {
    "mse": torch.nn.MSELoss,
    "huber": torch.nn.HuberLoss,
    "l1": torch.nn.L1Loss,
}


class LSTMModule(LightningModuleBase):
    def __init__(
        self,
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
        validate_every_n_epochs: int = 50,
        log_grad_norm: bool = False,
        criterion: torch.nn.Module | LOSS_FN = "mse",
        lr_scheduler: str | LR_CALL_TYPE | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
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
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler=lr_scheduler,
        )
        ignore = ["lr_scheduler"]
        if isinstance(criterion, torch.nn.Module):
            ignore.append("criterion")
        self.save_hyperparameters(ignore=ignore)

        self._val_hidden: list[HIDDEN_STATE | None] = []  # type: ignore[assignment]

        self.model = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

        self.criterion = (
            criterion
            if isinstance(criterion, torch.nn.Module)
            else CRITERION[criterion]()
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        return_states: typing.Literal[False] = False,
        hidden_state: HIDDEN_STATE | None = None,
    ) -> torch.Tensor:
        ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        return_states: typing.Literal[True],
        hidden_state: HIDDEN_STATE | None = None,
    ) -> tuple[torch.Tensor, HIDDEN_STATE]:
        ...

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
        hidden_state: HIDDEN_STATE | None = None,
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
        else:
            return y_hat

    @classmethod
    def parse_config_kwargs(
        cls, config: LSTMConfig, **kwargs: typing.Any  # type: ignore[override]
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        default_kwargs = dict(
            num_layers=config.num_layers,
            sequence_length=config.seq_len,
            hidden_dim=config.hidden_size,
            hidden_dim_fc=config.hidden_size_fc,
            dropout=config.dropout,
            lr=config.lr,
            max_epochs=config.num_epochs,
            optimizer=config.optimizer,
            optimizer_kwargs=config.optimizer_kwargs,
            validate_every_n_epochs=config.validate_every,
            log_grad_norm=config.log_grad_norm,
            criterion=config.loss_fn,
            lr_scheduler=config.lr_scheduler,
            lr_scheduler_interval=config.lr_scheduler_interval,
        )

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

        hidden = ops.detach(hidden)

        loss = self.criterion(output, target)

        # also compute MSE and MAE
        loss1 = torch.nn.MSELoss()(output, target)
        loss2 = torch.nn.L1Loss()(output, target)

        loss_dict = {
            "loss": loss,
            "loss_MSE": loss1,
            "loss_MAE": loss2,
        }

        return loss_dict, output, hidden  # type: ignore[misc]

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> torch.Tensor:
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
    ) -> STEP_OUTPUT:
        if dataloader_idx >= len(self._val_hidden):
            self._val_hidden.append(None)

        prev_hidden = self._val_hidden[dataloader_idx]

        loss, model_output, hidden = self.common_test_step(  # type: ignore[type-var]
            batch, batch_idx, prev_hidden
        )

        self._val_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        self.common_log_step(loss, "validation")

        return typing.cast(
            STEP_OUTPUT,
            loss
            | {
                "state": ops.to_cpu(ops.detach(hidden)),
                "output": ops.to_cpu(ops.detach(model_output)),
            },
        )
