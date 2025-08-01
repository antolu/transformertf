from __future__ import annotations

import collections.abc
import logging
import typing
from typing import NotRequired

import torch

from ...data import TimeSeriesSample
from ...utils import ops
from .._base_module import DEFAULT_LOGGING_METRICS, LightningModuleBase, MetricLiteral
from . import typing as bwt
from ._loss import BoucWenLoss
from ._model import BWLSTM1Model, BWLSTM2Model, BWLSTM3Model

log = logging.getLogger(__name__)


class PredictOutput(typing.TypedDict):
    """
    The output of a prediction step by the :class:`BWLSTM1`, :class:`BWLSTM2`
    or :class:`BWLSTM3` lightning modules.
    """

    output: bwt.BWLSTMOutput
    state: bwt.BWLSTMStates
    point_prediction: torch.Tensor


class StepOutput(PredictOutput):
    """
    The output of a training, validation or test step by the
    :class:`BWLSTM1`, :class:`BWLSTM2` or :class:`BWLSTM3` lightning modules.

    The :attr:`loss3`, :attr:`loss4` and :attr:`loss5` keys are only
    present when using the :class:`BWLSTM2` or :class:`BWLSTM3` modules.
    """

    loss: torch.Tensor
    """ The sum of all losses. """

    loss1: torch.Tensor
    loss2: torch.Tensor
    loss3: NotRequired[torch.Tensor]
    loss4: NotRequired[torch.Tensor]
    loss5: NotRequired[torch.Tensor]


class BWLSTMBase(LightningModuleBase):
    """
    Base class for the :class:`BWLSTM1`, :class:`BWLSTM2` and :class:`BWLSTM3`
    lightning modules, defining common methods for training, validation, testing
    and prediction steps, as well as handling hidden states and logging.
    """

    def __init__(
        self,
        *,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ) -> None:
        super().__init__()

        # default assume we have one dataloader only
        self._val_hidden: bwt.HiddenStateNone = [None]  # type: ignore[assignment]
        self._test_hidden: bwt.HiddenStateNone = [None]  # type: ignore[assignment]
        self._predict_hidden: bwt.HiddenStateNone = [None]  # type: ignore[assignment]

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
        hidden_state: bwt.BWLSTMStates | None = None,
    ) -> tuple[dict[str, torch.Tensor], bwt.BWLSTMOutput, bwt.BWLSTMStates]:
        """
        Common test step for validation and test steps. This method is used
        by the :meth:`validation_step` and :meth:`test_step` methods. It
        computes the model output and loss, and logs the loss.

        If the model output has a batch dimension of 1, it is removed.

        Parameters
        ----------
        batch : TimeSeriesSample
            The batch of data.
        batch_idx : int
            The batch index.
        hidden_state : HiddenState | None
            The hidden states. If `None`, the hidden states are initialized
            by the model.

        Returns
        -------
        dict[str, torch.Tensor], BoucWenOutput, BoucWenStates
            The losses, model output and hidden states.
        """
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
        """
        The training step for the :class:`BWLSTM1`, :class:`BWLSTM2` and
        :class:`BWLSTM3` lightning modules. This method computes the model
        output and loss, and logs the loss.

        No hidden states are passed to the model, as they are initialized
        by the model.

        Parameters
        ----------
        batch : TimeSeriesSample
            The batch of data.
        batch_idx : int
            The batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            The losses, with the key "loss".
        """
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

        return losses | {"output": model_output}

    def on_validation_batch_start(
        self, batch: TimeSeriesSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Initialize hidden states for the validation step."""
        if dataloader_idx >= len(self._val_hidden):
            self._val_hidden.append(None)

    def on_test_batch_start(
        self, batch: TimeSeriesSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Initialize hidden states for the test step."""
        if dataloader_idx >= len(self._test_hidden):
            self._test_hidden.append(None)

    def on_predict_batch_start(
        self, batch: TimeSeriesSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Initialize hidden states for the prediction step."""
        if dataloader_idx >= len(self._predict_hidden):
            self._predict_hidden.append(None)

    def validation_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        """
        The validation step for the :class:`BWLSTM1`, :class:`BWLSTM2` and
        :class:`BWLSTM3` lightning modules. This method computes the model
        output and loss, and logs the loss.

        The hidden states are passed to the model, and updated after the step.

        Parameters
        ----------
        batch : TimeSeriesSample
            The batch of data.
        batch_idx : int
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        StepOutput
            The losses, model output and hidden states.
        """
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
        """
        The test step for the :class:`BWLSTM1`, :class:`BWLSTM2` and
        :class:`BWLSTM3` lightning modules. This method computes the model
        output and loss, and logs the loss.

        The hidden states are passed to the model, and updated after the step.

        Parameters
        ----------
        batch : TimeSeriesSample
            The batch of data
        batch_idx : int
            The batch index.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        StepOutput
            The losses, model output and hidden states.
        """
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
        prev_hidden: bwt.BWLSTMStates | dict = (
            self._predict_hidden[dataloader_idx] or {}
        )

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
        """
        Add the point prediction to the outputs dictionary. The point
        prediction is ordinarily the goal of the model, which is fitted
        using the PINN loss function.

        Parameters
        ----------
        outputs : StepOutput
            The outputs dictionary, created by the :meth:`training_step`,
            :meth:`validation_step` or :meth:`test_step` methods.
        """
        outputs["point_prediction"] = self.criterion.point_prediction(outputs["output"])

    def on_train_batch_end(
        self,
        outputs: StepOutput,  # type: ignore[override]
        batch: TimeSeriesSample,  # type: ignore[override]
        batch_idx: int,
    ) -> None:
        """Add the point prediction to the outputs."""
        self.add_point_prediction(outputs)

        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        outputs: StepOutput,  # type: ignore[override]
        batch: TimeSeriesSample,  # type: ignore[override]
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save the hidden states."""
        self.add_point_prediction(outputs)
        self._val_hidden[dataloader_idx] = ops.detach(outputs["state"])

        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        outputs: StepOutput,  # type: ignore[override]
        batch: TimeSeriesSample,  # type: ignore[override]
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save the hidden states."""
        self.add_point_prediction(outputs)
        self._test_hidden[dataloader_idx] = ops.detach(outputs["state"])

        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)


class BWLSTM1(BWLSTMBase):
    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 350,
        n_dim_fc: int | None = None,
        dropout: float = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`BWLSTM1Model`, and returns
        the model output and losses corresponding to the :class:`BWLSTM1Model` model.

        Parameters
        ----------
        num_layers : int
            The number of LSTM layers.
        n_dim_model : int
            The number of hidden units in each LSTM layer.
        n_dim_fc : int, optional
            The number of hidden units in the fully connected layer. If `None`,
            this is set to `n_dim_model // 2`.
        dropout : float
            The dropout probability.
        loss_weights : BoucWenLoss.LossWeights, optional
            The loss weights to be used with the :class:`BoucWenLoss` loss function.
            If `None`, the default loss weights are used from the loss function.
        log_grad_norm : bool
            Whether to log the gradient norm at each step.
        """
        super().__init__()
        self.criterion = BoucWenLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["loss_weights"])

        self.bwlstm1 = BWLSTM1Model(
            num_layers=num_layers,
            d_model=d_model,
            n_dim_fc=n_dim_fc,
            dropout=dropout,
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> bwt.BWOutput1: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[bwt.BWOutput1, bwt.BWState1]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> bwt.BWOutput1 | tuple[bwt.BWOutput1, bwt.BWState1]:
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
        BWOuput1 | tuple[BWOutput1, BWState1]
            The model output.
        """
        return self.bwlstm1(x, hx=hx, return_states=return_states)


class BWLSTM2(BWLSTMBase):
    def __init__(
        self,
        num_layers: int | tuple[int, int] = 3,
        d_model: int | tuple[int, int] = 350,
        n_dim_fc: int | tuple[int, int] | None = None,
        dropout: float | tuple[float, float] = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`BWLSTM1Model` and
        :class:`BWLSTM2Model`, and returns the model output and losses
        corresponding to the :class:`BWLSTM1Model` and :class:`BWLSTM2Model` models.

        For each class parameter, if a single value is provided, it is used for
        all models. If a tuple of values is provided, the values are used for
        the corresponding models.

        Parameters
        ----------
        num_layers : int | tuple[int, int]
            The number of LSTM layers.
        n_dim_model : int | tuple[int, int]
            The number of hidden units in each LSTM layer.
        n_dim_fc : int | tuple[int, int], optional
            The number of hidden units in the fully connected layer. If `None`,
            this is set to `n_dim_model // 2`.
        dropout : float | tuple[float, float]
            The dropout probability.
        loss_weights : BoucWenLoss.LossWeights, optional
            The loss weights to be used with the :class:`BoucWenLoss` loss function.
            If `None`, the default loss weights are used from the loss function.
        log_grad_norm : bool
            Whether to log the gradient norm at each step.
        logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
            Container of metric names to compute and log during training, validation, and testing.
            If empty, no additional metrics will be logged (only the loss from the criterion).
            Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE".
        """
        super().__init__()
        self.criterion = BoucWenLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["loss_weights"])

        num_layers_ = _parse_tuple_or_int(num_layers, 2)
        d_model_ = _parse_tuple_or_int(d_model, 2)
        if n_dim_fc is None:
            n_dim_fc_ = tuple(dim // 2 for dim in d_model_)
        else:
            n_dim_fc_ = _parse_tuple_or_int(n_dim_fc, 2)

        dropout_ = _parse_tuple_or_int(dropout, 2)

        self.bwlstm1 = BWLSTM1Model(
            num_layers=num_layers_[0],
            d_model=d_model_[0],
            n_dim_fc=n_dim_fc_[0],
            dropout=dropout_[0],
        )
        self.bwlstm2 = BWLSTM2Model(
            num_layers=num_layers_[1],
            d_model=d_model_[1],
            n_dim_fc=n_dim_fc_[1],
            dropout=dropout_[1],
        )

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        hx2: bwt.LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> bwt.BWOutput12: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        hx2: bwt.LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[bwt.BWOutput12, bwt.BWState2]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        hx2: bwt.LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> bwt.BWOutput12 | tuple[bwt.BWOutput2, bwt.BWState2]:
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
        BWOuput12 | tuple[BWOutput12, BWState12]
            Model output. If `return_states` is `True`,
            the hidden states are also returned.
        """
        out, hx = self.bwlstm1(x, hx=hx, return_states=True)

        out2, hx2 = self.bwlstm2(x, out["z"], hx=hx2, return_states=True)

        out = out | out2
        hidden = {"hx": hx, "hx2": hx2}

        if return_states:
            return typing.cast(bwt.BWOutput12, out), typing.cast(bwt.BWState2, hidden)
        return typing.cast(bwt.BWOutput12, out)


class BWLSTM3(BWLSTMBase):
    def __init__(
        self,
        n_features: int = 1,
        num_layers: int | tuple[int, int, int] = 3,
        d_model: int | tuple[int, int, int] = 350,
        n_dim_fc: int | tuple[int, int, int] | None = None,
        dropout: float | tuple[float, float, float] = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        """
        This module implements a PyTorch Lightning module for hysteresis
        modeling. The module wraps an instance of :class:`PhyLSTM` and defines
        the training, validation, testing and prediction steps.

        The model parameters are saved to the model directory by Lightning.

        :param num_layers: The number of LSTM layers.
        :param d_model: The number of hidden units in each LSTM layer.
        :param n_dim_fc: The number of hidden units in the fully connected layer.
        :param dropout: The dropout probability.
        :param lr: The optimizer learning rate. This may be set to "auto"
                   for use with the Lightning Learning Rate Finder.
        :param loss_weights: The loss function to be used.
        :param logging_metrics: Container of metric names to compute and log during training, validation, and testing.
        """
        super().__init__()
        self.criterion: BoucWenLoss = BoucWenLoss(loss_weights=loss_weights)
        self.save_hyperparameters(ignore=["loss_weights"])

        num_layers_ = _parse_tuple_or_int(num_layers, 3)
        n_dim_model_ = _parse_tuple_or_int(d_model, 3)

        if n_dim_fc is None:
            n_dim_fc_ = tuple(dim // 2 for dim in n_dim_model_)
        else:
            n_dim_fc_ = _parse_tuple_or_int(n_dim_fc, 3)
        dropout_ = _parse_tuple_or_int(dropout, 3)

        self.bwlstm1 = BWLSTM1Model(
            n_features=n_features,
            num_layers=num_layers_[0],
            d_model=n_dim_model_[0],
            n_dim_fc=n_dim_fc_[0],
            dropout=dropout_[0],
        )
        self.bwlstm2 = BWLSTM2Model(
            num_layers=num_layers_[1],
            d_model=n_dim_model_[1],
            n_dim_fc=n_dim_fc_[1],
            dropout=dropout_[1],
        )
        self.bwlstm3 = BWLSTM3Model(
            num_layers=num_layers_[2],
            d_model=n_dim_model_[2],
            n_dim_fc=n_dim_fc_[2],
            dropout=dropout_[2],
        )

    # def maybe_compile_model(self) -> None:
    #     if (
    #         "compile_model" in self.hparams
    #         and self.hparams["compile_model"]
    #         and hasattr(self, "bwlstm1")
    #         and hasattr(self, "bwlstm2")
    #         and hasattr(self, "bwlstm3")
    #     ):
    #         self.bwlstm1 = torch.compile(self.bwlstm1)
    #         self.bwlstm2 = torch.compile(self.bwlstm2)
    #         self.bwlstm3 = torch.compile(self.bwlstm3)
    #
    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        hx2: bwt.LSTMState | None = None,
        hx3: bwt.LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> bwt.BWOutput3: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        hx2: bwt.LSTMState | None = None,
        hx3: bwt.LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[bwt.BWOutput3, bwt.BWState3]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: bwt.LSTMState | None = None,
        hx2: bwt.LSTMState | None = None,
        hx3: bwt.LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> bwt.BWOutput3 | tuple[bwt.BWOutput3, bwt.BWState3]:
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
            return typing.cast(bwt.BWOutput123, out), typing.cast(bwt.BWState3, hidden)
        return typing.cast(bwt.BWOutput3, out)


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
