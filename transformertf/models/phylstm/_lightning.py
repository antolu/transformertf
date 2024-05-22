from __future__ import annotations

import typing
from typing import NotRequired

import torch

from ...data import TimeSeriesSample
from ...utils import ops
from .._base_module import LightningModuleBase
from ._loss import PhyLSTMLoss
from ._model import PhyLSTM1Model, PhyLSTM2Model, PhyLSTM3Model
from ._output import (
    PhyLSTM1Output,
    PhyLSTM1States,
    PhyLSTM2Output,
    PhyLSTM2States,
    PhyLSTM3Output,
    PhyLSTM3States,
)

HiddenState: typing.TypeAlias = PhyLSTM1States | PhyLSTM2States | PhyLSTM3States
HiddenStateNone: typing.TypeAlias = (
    list[PhyLSTM1States | None]
    | list[PhyLSTM2States | None]
    | list[PhyLSTM3States | None]
)
PhyLSTMOutput: typing.TypeAlias = PhyLSTM1Output | PhyLSTM2Output | PhyLSTM3Output


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    loss1: torch.Tensor
    loss2: torch.Tensor
    loss3: NotRequired[torch.Tensor]
    loss4: NotRequired[torch.Tensor]
    loss5: NotRequired[torch.Tensor]
    output: PhyLSTMOutput
    state: HiddenState


class PredictOutput(typing.TypedDict):
    output: PhyLSTMOutput
    state: HiddenState


class PhyLSTM(LightningModuleBase):
    def __init__(
        self,
        num_layers: int | tuple[int, ...] = 3,
        sequence_length: int = 500,
        hidden_dim: int | tuple[int, ...] = 350,
        hidden_dim_fc: int | tuple[int, ...] | None = None,
        dropout: float | tuple[float, ...] = 0.2,
        phylstm: typing.Literal[1, 2, 3] | None = 3,
        loss_weights: PhyLSTMLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`PhyLSTM` and defines
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
        :param datamodule: The data module to be get the dataloaders from,
            if a Trainer is not attached.
        """
        super().__init__()
        self.criterion = PhyLSTMLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["criterion", "loss_weights"])

        self._val_hidden: HiddenStateNone = []  # type: ignore[assignment]
        self._test_hidden: HiddenStateNone = []  # type: ignore[assignment]
        self._predict_hidden: HiddenStateNone = []  # type: ignore[assignment]

        model_cls: type[PhyLSTM1Model | PhyLSTM2Model | PhyLSTM3Model]
        if phylstm == 1:
            model_cls = PhyLSTM1Model
        elif phylstm == 2:
            model_cls = PhyLSTM2Model
        elif phylstm == 3 or phylstm is None:
            model_cls = PhyLSTM3Model
        else:
            msg = "phylstm must be 1, 2 or 3"
            raise ValueError(msg)

        self.model = model_cls(
            num_layers=num_layers,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: HiddenState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> PhyLSTMOutput: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: HiddenState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[PhyLSTMOutput, HiddenState]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: HiddenState | None = None,
        *,
        return_states: bool = False,
    ) -> PhyLSTMOutput | tuple[PhyLSTMOutput, HiddenState]:
        """
        Forward pass through the model.
        Rescales the output to the target scale if provided.

        :param x: The input sequence.
        :param hidden: The hidden states.
        :param target_scale: The target scale.
        :param return_states: Whether to return the hidden states.
        :return: The model output.
        """
        return self.model(x, hx=hx, return_states=return_states)

    def on_validation_epoch_start(self) -> None:
        """Reset the hidden states"""
        self._val_hidden = [None]  # type: ignore[assignment]

        super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        """Reset the hidden states"""
        self._test_hidden = [None]  # type: ignore[assignment]

        super().on_test_epoch_start()

    def on_predict_epoch_start(self) -> None:
        """Reset the hidden states"""
        self._predict_hidden = [None]  # type: ignore[assignment]

        super().on_predict_epoch_start()

    def common_test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        hidden_state: HiddenState | None = None,
    ) -> tuple[dict[str, torch.Tensor], PhyLSTMOutput, HiddenState]:
        target = batch.get("target")
        assert target is not None

        hidden: HiddenState
        output, hidden = self.forward(
            batch["input"],
            hx=hidden_state,
            return_states=True,
        )

        hidden = ops.detach(hidden)  # type: ignore[type-var]

        _, losses = self.criterion(output, target, return_all=True)

        # remove batch dimension
        assert output["z"].shape[0] == 1
        for key in output:
            output[key] = output[key].squeeze(0)  # type: ignore[literal-required]

        return losses, output, hidden  # type: ignore[misc]

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        target = batch.get("target")
        batch.get("target_scale")
        assert target is not None

        model_output = self.forward(batch["input"])

        _, losses = self.criterion(model_output, target, return_all=True)

        self.common_log_step(losses, "train")

        return losses

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
                "state": ops.to_cpu(ops.detach(hidden)),  # type: ignore[type-var]
                "output": ops.to_cpu(ops.detach(model_output)),  # type: ignore[type-var]
            },
        )

    def test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        if dataloader_idx >= len(self._test_hidden):
            self._test_hidden.append(None)

        prev_hidden = self._test_hidden[dataloader_idx]

        loss, model_output, hidden = self.common_test_step(  # type: ignore[type-var]
            batch, batch_idx, prev_hidden
        )

        self._test_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        self.common_log_step(loss, "test")

        return typing.cast(
            StepOutput,
            loss
            | {
                "state": ops.to_cpu(ops.detach(hidden)),  # type: ignore[type-var]
                "output": ops.to_cpu(ops.detach(model_output)),  # type: ignore[type-var]
            },
        )

    def predict_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PredictOutput:
        if dataloader_idx >= len(self._predict_hidden):
            self._predict_hidden.append(None)

        prev_hidden = self._predict_hidden[dataloader_idx]

        model_output, hidden = self.forward(
            batch["input"], hx=prev_hidden, return_states=True
        )
        self._predict_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        return typing.cast(
            PredictOutput,
            {
                "state": ops.to_cpu(ops.detach(hidden)),  # type: ignore[type-var]
                "output": ops.to_cpu(ops.detach(model_output)),  # type: ignore[type-var]
            },
        )
