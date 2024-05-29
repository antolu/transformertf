from __future__ import annotations

import logging
import typing
from typing import NotRequired

import torch

from ...data import TimeSeriesSample
from ...utils import ops
from .._base_module import LightningModuleBase
from . import typing as t
from ._loss import BoucWenLoss
from ._model import BWLSTM1Model, BWLSTM2Model, BWLSTM3Model

log = logging.getLogger(__name__)


class PredictOutput(typing.TypedDict):
    output: t.BWLSTMOutput
    state: t.BWLSTMStates
    point_prediction: torch.Tensor


class StepOutput(PredictOutput):
    loss: torch.Tensor
    loss1: torch.Tensor
    loss2: torch.Tensor
    loss3: NotRequired[torch.Tensor]
    loss4: NotRequired[torch.Tensor]
    loss5: NotRequired[torch.Tensor]


class BWLSTMBase(LightningModuleBase):
    def __init__(self) -> None:
        super().__init__()

        # default assume we have one dataloader only
        self._val_hidden: t.HiddenStateNone = [None]  # type: ignore[assignment]
        self._test_hidden: t.HiddenStateNone = [None]  # type: ignore[assignment]
        self._predict_hidden: t.HiddenStateNone = [None]  # type: ignore[assignment]

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
        hidden_state: t.BWLSTMStates | None = None,
    ) -> tuple[dict[str, torch.Tensor], t.BWLSTMOutput, t.BWLSTMStates]:
        output, hidden = self.forward(
            batch["input"],
            **(hidden_state or {}),
            return_states=True,
        )

        try:
            _, losses = self.criterion(output, batch["target"], return_all=True)
        except KeyError:
            losses = {}

        # remove batch dimension
        if output["z"].shape[0] == 1:
            for key in output:
                output[key] = output[key].squeeze(0)  # type: ignore[literal-required]

        return losses, output, hidden  # type: ignore[misc]

    def training_step(
        self, batch: TimeSeriesSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        try:
            target = batch["target"]
        except KeyError as e:
            msg = (
                "The batch must contain a target key. "
                "This is probably due to using a dataset without targets "
                "(e.g. test or predict)."
            )
            raise ValueError(msg) from e

        model_output = self.forward(batch["input"])

        _, losses = self.criterion(model_output, target, return_all=True)
        self.common_log_step(losses, "train")

        return losses

    def on_validation_batch_start(
        self, batch: TimeSeriesSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if dataloader_idx >= len(self._val_hidden):
            self._val_hidden.append(None)

    def on_test_batch_start(
        self, batch: TimeSeriesSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if dataloader_idx >= len(self._test_hidden):
            self._test_hidden.append(None)

    def on_predict_batch_start(
        self, batch: TimeSeriesSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if dataloader_idx >= len(self._predict_hidden):
            self._predict_hidden.append(None)

    def validation_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        prev_hidden = self._val_hidden[dataloader_idx]

        loss, model_output, hidden = self.common_test_step(
            batch, batch_idx, prev_hidden
        )

        self.common_log_step(loss, "validation")

        return typing.cast(
            StepOutput,
            loss
            | {
                "state": hidden,
                "output": model_output,
            },
        )

    def test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        prev_hidden = self._test_hidden[dataloader_idx]

        loss, output, hidden = self.common_test_step(  # type: ignore[type-var]
            batch, batch_idx, prev_hidden
        )

        self.common_log_step(loss, "test")

        return typing.cast(
            StepOutput,
            loss
            | {
                "state": hidden,
                "output": output,
            },
        )

    def predict_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PredictOutput:
        prev_hidden: t.BWLSTMStates | dict = self._predict_hidden[dataloader_idx] or {}

        output, hidden = self.forward(
            batch["input"], hx=prev_hidden.get("hx"), return_states=True
        )
        self._predict_hidden[dataloader_idx] = hidden

        return typing.cast(
            PredictOutput,
            {
                "state": hidden,
                "output": output,
            },
        )

    def add_point_prediction(self, outputs: StepOutput) -> None:
        outputs["point_prediction"] = self.criterion.point_prediction(outputs["output"])

    def on_train_batch_end(
        self,
        outputs: StepOutput,  # type: ignore[override]
        batch: TimeSeriesSample,  # type: ignore[override]
        batch_idx: int,
    ) -> None:
        self.add_point_prediction(outputs)

    def on_validation_batch_end(
        self,
        outputs: StepOutput,  # type: ignore[override]
        batch: TimeSeriesSample,  # type: ignore[override]
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._val_hidden[dataloader_idx] = ops.detach(outputs["state"])

    def on_test_batch_end(
        self,
        outputs: StepOutput,  # type: ignore[override]
        batch: TimeSeriesSample,  # type: ignore[override]
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._test_hidden[dataloader_idx] = ops.detach(outputs["state"])


class BWLSTM1(BWLSTMBase):
    def __init__(
        self,
        num_layers: int,
        n_dim_model: int = 350,
        n_dim_fc: int | None = None,
        dropout: float = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`PhyLSTM` and defines
        the training, validation, testing and prediction steps.

        The model parameters are saved to the model directory by Lightning.

        :param num_layers: The number of LSTM layers.
        :param n_dim_model: The number of hidden units in each LSTM layer.
        :param n_dim_fc: The number of hidden units in the fully connected layer.
        :param dropout: The dropout probability.
        :param lr: The optimizer learning rate. This may be set to "auto"
                   for use with the Lightning Learning Rate Finder.
        :param loss_weights: The loss function to be used.
        """
        super().__init__()
        self.criterion = BoucWenLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["loss_weights"])

        self.bwlstm1 = BWLSTM1Model(
            num_layers=num_layers,
            n_dim_model=n_dim_model,
            n_dim_fc=n_dim_fc,
            dropout=dropout,
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> t.BWOutput1: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[t.BWOutput1, t.BWState1]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> t.BWOutput1 | tuple[t.BWOutput1, t.BWState1]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence. The shape is (seq_len, batch_size, num_features).
        hx : HiddenState | None
            The hidden states.
        return_states : bool
            Whether to return the hidden states.

        Returns
        -------
        BoucWenOutput1 | tuple[BoucWenOutput1, BoucWenStates1]
            The model output.
        """
        return self.bwlstm1(x, hx=hx, return_states=return_states)


class BWLSTM2(BWLSTMBase):
    def __init__(
        self,
        num_layers: int | tuple[int, int] = 3,
        n_dim_model: int | tuple[int, int] = 350,
        n_dim_fc: int | tuple[int, int] | None = None,
        dropout: float | tuple[float, float] = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`PhyLSTM` and defines
        the training, validation, testing and prediction steps.

        The model parameters are saved to the model directory by Lightning.

        :param num_layers: The number of LSTM layers.
        :param n_dim_model: The number of hidden units in each LSTM layer.
        :param n_dim_fc: The number of hidden units in the fully connected layer.
        :param dropout: The dropout probability.
        :param lr: The optimizer learning rate. This may be set to "auto"
                   for use with the Lightning Learning Rate Finder.
        :param loss_weights: The loss function to be used.
        """
        super().__init__()
        self.criterion = BoucWenLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["loss_weights"])

        num_layers_ = _parse_tuple_or_int(num_layers, 2)
        n_dim_model_ = _parse_tuple_or_int(n_dim_model, 2)
        if n_dim_fc is None:
            n_dim_fc_ = tuple([dim // 2 for dim in n_dim_model_])
        else:
            n_dim_fc_ = _parse_tuple_or_int(n_dim_fc, 2)

        dropout_ = _parse_tuple_or_int(dropout, 2)

        self.bwlstm1 = BWLSTM1Model(
            num_layers=num_layers_[0],
            n_dim_model=n_dim_model_[0],
            n_dim_fc=n_dim_fc_[0],
            dropout=dropout_[0],
        )
        self.bwlstm2 = BWLSTM2Model(
            num_layers=num_layers_[1],
            n_dim_model=n_dim_model_[1],
            n_dim_fc=n_dim_fc_[1],
            dropout=dropout_[1],
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        hx2: t.LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> t.BWOutput12: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        hx2: t.LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[t.BWOutput12, t.BWState2]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        hx2: t.LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> t.BWOutput12 | tuple[t.BWOutput2, t.BWState2]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence. The shape is (seq_len, batch_size, num_features).
        hx : LSTMState | None
            The hidden states for the first :class:`BWLSTM1Model`.
        hx2 : LSTMState | None
            The hidden states for the second :class:`BWLSTM2Model`.
        return_states : bool
            Whether to return the hidden states.

        Returns
        -------
        BoucWenOutput2 | tuple[BoucWenOutput2, BoucWenStates2]
            Model output. If `return_states` is `True`,
            the hidden states are also returned.
        """
        out, hx = self.bwlstm1(x, hx=hx, return_states=True)

        out2, hx2 = self.bwlstm2(x, out["z"], hx=hx2, return_states=True)

        out = out | out2
        hidden = {"hx": hx, "hx2": hx2}

        if return_states:
            return typing.cast(t.BWOutput12, out), typing.cast(t.BWState2, hidden)
        return typing.cast(t.BWOutput12, out)


class BWLSTM3(BWLSTMBase):
    def __init__(
        self,
        num_layers: int | tuple[int, int, int] = 3,
        n_dim_model: int | tuple[int, int, int] = 350,
        n_dim_fc: int | tuple[int, int, int] | None = None,
        dropout: float | tuple[float, float, float] = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`PhyLSTM` and defines
        the training, validation, testing and prediction steps.

        The model parameters are saved to the model directory by Lightning.

        :param num_layers: The number of LSTM layers.
        :param n_dim_model: The number of hidden units in each LSTM layer.
        :param n_dim_fc: The number of hidden units in the fully connected layer.
        :param dropout: The dropout probability.
        :param lr: The optimizer learning rate. This may be set to "auto"
                   for use with the Lightning Learning Rate Finder.
        :param loss_weights: The loss function to be used.
        """
        super().__init__()
        self.criterion = BoucWenLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["loss_weights"])

        num_layers_ = _parse_tuple_or_int(num_layers, 3)
        n_dim_model_ = _parse_tuple_or_int(n_dim_model, 3)

        if n_dim_fc is None:
            n_dim_fc_ = tuple([dim // 2 for dim in n_dim_model_])
        else:
            n_dim_fc_ = _parse_tuple_or_int(n_dim_fc, 3)
        dropout_ = _parse_tuple_or_int(dropout, 3)

        self.bwlstm1 = BWLSTM1Model(
            num_layers=num_layers_[0],
            n_dim_model=n_dim_model_[0],
            n_dim_fc=n_dim_fc_[0],
            dropout=dropout_[0],
        )
        self.bwlstm2 = BWLSTM2Model(
            num_layers=num_layers_[1],
            n_dim_model=n_dim_model_[1],
            n_dim_fc=n_dim_fc_[1],
            dropout=dropout_[1],
        )
        self.bwlstm3 = BWLSTM3Model(
            num_layers=num_layers_[2],
            n_dim_model=n_dim_model_[2],
            n_dim_fc=n_dim_fc_[2],
            dropout=dropout_[2],
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        hx2: t.LSTMState | None = None,
        hx3: t.LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> t.BWOutput3: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        hx2: t.LSTMState | None = None,
        hx3: t.LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[t.BWOutput3, t.BWState3]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: t.LSTMState | None = None,
        hx2: t.LSTMState | None = None,
        hx3: t.LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> t.BWOutput3 | tuple[t.BWOutput3, t.BWState3]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence. The shape is (seq_len, batch_size, num_features).
        hx : LSTMState | None
            The hidden states for the first :class:`BWLSTM1Model`.
        hx2 : LSTMState | None
            The hidden states for the second :class:`BWLSTM2Model`.
        hx3 : LSTMState | None
            The hidden states for the third :class:`BWLSTM3Model`.
        return_states : bool
            Whether to return the hidden states.

        Returns
        -------
        BoucWenOutput3 | tuple[BoucWenOutput3, BoucWenStates3]
            Model output. If `return_states` is `True`,
            the hidden states are also returned.
        """
        out, hx = self.bwlstm1(x, hx=hx, return_states=True)

        out2, hx2 = self.bwlstm2(x, out["z"], hx=hx2, return_states=True)

        out3, hx3 = self.bwlstm3(out["z"], out2["dz_dt"], hx=hx3, return_states=True)

        out = out | out2 | out3
        hidden = {"hx": hx, "hx2": hx2, "hx3": hx3}

        if return_states:
            return typing.cast(t.BWOutput123, out), typing.cast(t.BWState3, hidden)
        return typing.cast(t.BWOutput3, out)


U = typing.TypeVar("U", int, float)


def _parse_tuple_or_int(value: U | tuple[U, ...], num_elements: int) -> tuple[U, ...]:
    if isinstance(value, tuple):
        if len(value) != num_elements:
            msg = f"Expected a tuple of length {num_elements}, got {len(value)}"
            raise ValueError(
                msg,
            )
        return value

    return tuple([value] * num_elements)
