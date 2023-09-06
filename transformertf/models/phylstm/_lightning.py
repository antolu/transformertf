from __future__ import annotations

import sys
import typing
from functools import partial

if sys.version_info >= (3, 10):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

import lightning as L
import torch

from ...data import TimeSeriesSample
from ...utils import ops
from .._base_module import LightningModuleBase
from ._loss import PhyLSTMLoss
from ._model import PhyLSTM1, PhyLSTM2, PhyLSTM3
from ._output import (
    PhyLSTM1Output,
    PhyLSTM1States,
    PhyLSTM2Output,
    PhyLSTM2States,
    PhyLSTM3Output,
    PhyLSTM3States,
)

if typing.TYPE_CHECKING:
    from ._config import PhyLSTMConfig

    SameType = typing.TypeVar("SameType", bound="PhyLSTMModule")


HIDDEN_STATE = typing.Union[PhyLSTM1States, PhyLSTM2States, PhyLSTM3States]
HIDDEN_STATES = typing.Union[
    list[typing.Union[PhyLSTM1States, None]],
    list[typing.Union[PhyLSTM2States, None]],
    list[typing.Union[PhyLSTM3States, None]],
]
PHYLSTM_OUTPUT = typing.Union[PhyLSTM1Output, PhyLSTM2Output, PhyLSTM3Output]

STEP_OUTPUT = typing.TypedDict(
    "STEP_OUTPUT",
    {
        "loss": torch.Tensor,
        "loss1": torch.Tensor,
        "loss2": torch.Tensor,
        "loss3": NotRequired[torch.Tensor],
        "loss4": NotRequired[torch.Tensor],
        "loss5": NotRequired[torch.Tensor],
        "output": PHYLSTM_OUTPUT,
        "state": HIDDEN_STATE,
    },
)
PREDICT_OUTPUT = typing.TypedDict(
    "PREDICT_OUTPUT", {"output": PHYLSTM_OUTPUT, "state": HIDDEN_STATE}
)


class PhyLSTMModule(LightningModuleBase):
    def __init__(
        self,
        num_layers: int | tuple[int, ...] = 3,
        sequence_length: int = 500,
        hidden_dim: int | tuple[int, ...] = 350,
        hidden_dim_fc: int | tuple[int, ...] | None = None,
        dropout: float | tuple[float, ...] = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        optimizer: str | None = None,
        optimizer_kwargs: dict | None = None,
        reduce_on_plateau_patience: int = 200,
        max_epochs: int = 1000,
        phylstm: typing.Literal[1, 2, 3] | None = 3,
        validate_every_n_epochs: int = 50,
        log_grad_norm: bool = False,
        criterion: PhyLSTMLoss | None = None,
        lr_scheduler: str
        | typing.Type[torch.optim.lr_scheduler.LRScheduler]
        | partial
        | None = None,
        lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch",
        datamodule: L.LightningDataModule | None = None,
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
        self.save_hyperparameters(
            ignore=["criterion", "lr_scheduler", "datamodule"]
        )

        self._val_hidden: HIDDEN_STATES = []  # type: ignore[assignment]
        self._test_hidden: HIDDEN_STATES = []  # type: ignore[assignment]
        self._predict_hidden: HIDDEN_STATES = []  # type: ignore[assignment]

        self.criterion: PhyLSTMLoss = criterion or PhyLSTMLoss()
        self._lr_scheduler = lr_scheduler
        self._datamodule = datamodule

        model_cls: typing.Type[PhyLSTM1] | typing.Type[PhyLSTM2] | typing.Type[
            PhyLSTM3
        ]
        if phylstm == 1:
            model_cls = PhyLSTM1
        elif phylstm == 2:
            model_cls = PhyLSTM2
        elif phylstm == 3 or phylstm is None:
            model_cls = PhyLSTM3
        else:
            raise ValueError("phylstm must be 1, 2 or 3")

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
        return_states: typing.Literal[False] = False,
        hidden_state: HIDDEN_STATE | None = None,
    ) -> PHYLSTM_OUTPUT:
        ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        return_states: typing.Literal[True],
        hidden_state: HIDDEN_STATE | None = None,
    ) -> tuple[PHYLSTM_OUTPUT, HIDDEN_STATE]:
        ...

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
        hidden_state: HIDDEN_STATE | None = None,
    ) -> PHYLSTM_OUTPUT | tuple[PHYLSTM_OUTPUT, HIDDEN_STATE]:
        """
        Forward pass through the model.
        Rescales the output to the target scale if provided.

        :param x: The input sequence.
        :param hidden: The hidden states.
        :param target_scale: The target scale.
        :param return_states: Whether to return the hidden states.
        :return: The model output.
        """
        return self.model(
            x, hidden_state=hidden_state, return_states=return_states
        )

    @classmethod
    def from_config(  # type: ignore[override]
        cls: typing.Type[SameType],
        config: PhyLSTMConfig,
        criterion: PhyLSTMLoss | None = None,
        lr_scheduler: str
        | typing.Type[torch.optim.lr_scheduler.LRScheduler]
        | partial
        | None = None,
        datamodule: L.LightningDataModule | None = None,
        **kwargs: typing.Any,
    ) -> SameType:
        new_kwargs = dict(
            phylstm=config.phylstm,
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
            criterion=criterion or PhyLSTMLoss.from_config(config),
            lr_scheduler=lr_scheduler or config.lr_scheduler,
            lr_scheduler_interval=config.lr_scheduler_interval,
            datamodule=datamodule,
        )

        kwargs.update(new_kwargs)
        return cls(**kwargs)

    @property
    def validation_outputs(self) -> list[STEP_OUTPUT]:
        return typing.cast(list[STEP_OUTPUT], self._val_outputs)

    @property
    def test_outputs(self) -> list[STEP_OUTPUT]:
        return typing.cast(list[STEP_OUTPUT], self._test_outputs)

    @property
    def predict_outputs(self) -> list[PREDICT_OUTPUT]:
        return typing.cast(list[PREDICT_OUTPUT], self._predict_outputs)

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
        hidden_state: HIDDEN_STATE | None = None,
    ) -> tuple[dict[str, torch.Tensor], PHYLSTM_OUTPUT, HIDDEN_STATE]:
        target = batch.get("target")
        target_scale = batch.get("target_scale")
        assert target is not None
        assert target_scale is not None

        hidden: HIDDEN_STATE
        output, hidden = self.forward(
            batch["input"],
            hidden_state=hidden_state,
            return_states=True,
        )

        hidden = ops.detach(hidden)

        _, losses = self.criterion(
            output, target, target_scale=target_scale, return_all=True
        )

        # remove batch dimension
        assert output["z"].shape[0] == 1
        for key in output.keys():
            output[key] = output[key].squeeze(0)  # type: ignore[literal-required]

        return losses, output, hidden  # type: ignore[misc]

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        target = batch.get("target")
        target_scale = batch.get("target_scale")
        assert target is not None
        assert target_scale is not None

        initial_state = typing.cast(PhyLSTM1States, {"lstm1": self.model.init_states(batch["initial"])})
        model_output = self.forward(batch["input"], hidden_state=initial_state)

        _, losses = self.criterion(
            model_output, target, target_scale=target_scale, return_all=True
        )

        self.common_log_step(losses, "train")

        return losses

    def validation_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        if dataloader_idx >= len(self._val_hidden):
            self._val_hidden.append(None)

        prev_hidden = self._val_hidden[dataloader_idx]
        if prev_hidden is None:
            prev_hidden = {"lstm1": self.model.init_states(batch["initial"])}

        loss, model_output, hidden = self.common_test_step(  # type: ignore[type-var]
            batch, batch_idx, prev_hidden
        )

        self._val_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        self.common_log_step(loss, "validation")

        return typing.cast(
            STEP_OUTPUT,
            loss | {
                "state": ops.to_cpu(ops.detach(hidden)),
                "output": ops.to_cpu(ops.detach(model_output)),
            },
        )

    def test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        if dataloader_idx >= len(self._test_hidden):
            self._test_hidden.append(None)

        prev_hidden = self._test_hidden[dataloader_idx]

        loss, model_output, hidden = self.common_test_step(  # type: ignore[type-var]
            batch, batch_idx, prev_hidden
        )

        self._test_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        self.common_log_step(loss, "test")

        return typing.cast(
            STEP_OUTPUT,
            loss | {
                "state": ops.to_cpu(ops.detach(hidden)),
                "output": ops.to_cpu(ops.detach(model_output)),
            },
        )

    def predict_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PREDICT_OUTPUT:
        if dataloader_idx >= len(self._predict_hidden):
            self._predict_hidden.append(None)

        prev_hidden = self._predict_hidden[dataloader_idx]

        model_output, hidden = self.forward(
            batch["input"], hidden_state=prev_hidden, return_states=True
        )
        self._predict_hidden[dataloader_idx] = hidden  # type: ignore[assignment]

        return typing.cast(
            PREDICT_OUTPUT,
            {
                "state": ops.to_cpu(ops.detach(hidden)),
                "output": ops.to_cpu(ops.detach(model_output)),
            },
        )
