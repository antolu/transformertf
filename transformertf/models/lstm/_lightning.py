from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import TimeSeriesSample
from ...nn import MLP, QuantileLoss
from ...utils import ops
from .._base_module import DEFAULT_LOGGING_METRICS, LightningModuleBase, MetricLiteral

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    state: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor | None]


class LSTM(LightningModuleBase):
    """
    LSTM model for time series forecasting with optional quantile prediction.

    This model implements a multi-layer LSTM network with a fully connected head
    for time series forecasting. It supports both point predictions and probabilistic
    forecasting via quantile regression. The model maintains hidden states across
    validation steps to handle sequential dependencies properly.

    The architecture consists of:
    1. Multi-layer LSTM for temporal modeling
    2. Fully connected layers for output projection
    3. Optional quantile loss for probabilistic forecasting

    Parameters
    ----------
    num_features : int, default=1
        Number of input features in the time series.
    num_layers : int, default=3
        Number of LSTM layers in the model. More layers can capture more
        complex temporal patterns but may lead to overfitting.
    d_model : int, default=350
        Hidden size of the LSTM layers. This controls the model capacity
        and computational requirements.
    n_layers_fc : int, default=1
        Number of fully connected layers in the output head. If > 1,
        creates an MLP with ReLU activations.
    d_fc : int or None, default=None
        Hidden dimension of the fully connected layers. If None, uses
        the same dimension as the LSTM hidden size.
    output_dim : int, default=1
        Output dimension. For quantile loss, this is automatically set
        to the number of quantiles.
    dropout : float, default=0.2
        Dropout probability applied to LSTM layers and fully connected layers.
    criterion : torch.nn.Module or None, default=None
        Loss function for training. If None, defaults to MSELoss.
        Use QuantileLoss for probabilistic forecasting.
    compile_model : bool, default=False
        Whether to compile the model using PyTorch 2.0 for improved performance.
    log_grad_norm : bool, default=False
        Whether to log gradient norms during training for debugging.

    Attributes
    ----------
    model : torch.nn.LSTM
        The main LSTM model.
    fc : torch.nn.Linear or MLP
        The fully connected output layer(s).
    criterion : torch.nn.Module
        The loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models import LSTM
    >>> from transformertf.data import TimeSeriesDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create LSTM model for point prediction
    >>> model = LSTM(
    ...     num_features=5,
    ...     num_layers=2,
    ...     d_model=128,
    ...     n_layers_fc=2,
    ...     d_fc=64,
    ...     output_dim=1,
    ...     dropout=0.1
    ... )
    >>>
    >>> # Create LSTM model for probabilistic forecasting
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = LSTM(
    ...     num_features=5,
    ...     num_layers=3,
    ...     d_model=256,
    ...     criterion=QuantileLoss(quantiles=quantiles),
    ...     compile_model=True
    ... )
    >>>
    >>> # Train with time series data
    >>> datamodule = TimeSeriesDataModule(...)
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
    >>>
    >>> # Generate predictions
    >>> predictions = trainer.predict(model, datamodule.test_dataloader())

    Notes
    -----
    **Hidden State Management:**

    The model maintains hidden states across validation steps to properly handle
    sequential dependencies. Hidden states are:
    - Reset at the beginning of each validation epoch
    - Maintained across batches within the same validation epoch
    - Automatically detached to prevent gradient accumulation

    **Architecture Details:**

    The LSTM model uses:
    - Batch-first convention for input tensors
    - Dropout between LSTM layers (when num_layers > 1)
    - Optional MLP head for complex output transformations
    - Automatic handling of quantile vs. point predictions

    **Input Requirements:**

    The model expects :class:`transformertf.data.TimeSeriesSample` with:
    - `input`: Input sequences (B, seq_len, num_features)
    - `target`: Target values (B, seq_len, 1) for training/validation

    **Quantile Support:**

    When using :class:`transformertf.nn.QuantileLoss`, the model automatically:
    - Sets output_dim to the number of quantiles
    - Extracts point predictions (median) from quantile outputs
    - Handles probabilistic forecasting workflow

    See Also
    --------
    LightningModuleBase : Base class for all models
    transformertf.data.TimeSeriesDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    transformertf.nn.MLP : Multi-layer perceptron for output head
    """

    def __init__(
        self,
        num_features: int = 1,
        num_layers: int = 3,
        d_model: int = 350,
        n_layers_fc: int = 1,
        d_fc: int | None = None,
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
        super().__init__()
        d_fc = d_fc or d_model

        self.criterion = criterion or torch.nn.MSELoss()
        self.save_hyperparameters(ignore=["criterion"])

        if isinstance(self.criterion, QuantileLoss):
            output_dim = self.criterion.num_quantiles
            self.hparams["output_dim"] = output_dim

        self._val_hidden: list[HIDDEN_STATE | None] = []
        self._test_hidden: list[HIDDEN_STATE | None] = []
        self._predict_hidden: list[HIDDEN_STATE | None] = []

        self.model = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        if n_layers_fc > 1:
            self.fc = MLP(
                input_dim=d_model,
                output_dim=output_dim,
                d_hidden=[d_fc] * (n_layers_fc - 1),
                activation="relu",
                dropout=dropout,
            )
        else:
            self.fc = torch.nn.Linear(d_model, output_dim)

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
        Forward pass through the LSTM model.

        This method processes the input sequence through the LSTM layers and
        output projection layers. It supports optional hidden state input for
        continuing sequences and can return updated hidden states.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, seq_len, num_features).
        hx : HIDDEN_STATE or None, default=None
            Initial hidden state tuple (h_0, c_0) where each has shape
            (num_layers, batch_size, d_model). If None, zero-initialized.
        return_states : bool, default=False
            Whether to return the final hidden states along with the output.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, HIDDEN_STATE]
            If return_states=False:
                Output tensor of shape (batch_size, seq_len, output_dim)
            If return_states=True:
                Tuple of (output, (h_n, c_n)) where:
                - output: (batch_size, seq_len, output_dim)
                - h_n: (num_layers, batch_size, d_model)
                - c_n: (num_layers, batch_size, d_model)

        Notes
        -----
        The forward pass consists of:
        1. LSTM processing: Input → LSTM layers → hidden representations
        2. Output projection: Hidden representations → FC layers → final output

        For quantile models, the output represents quantile predictions where
        each output dimension corresponds to a different quantile level.

        Examples
        --------
        >>> model = LSTM(num_features=3, d_model=64, output_dim=1)
        >>> x = torch.randn(32, 100, 3)  # batch_size=32, seq_len=100
        >>>
        >>> # Simple forward pass
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 100, 1])
        >>>
        >>> # Forward pass with state management
        >>> output, (h_n, c_n) = model(x, return_states=True)
        >>> # Continue with next sequence using previous states
        >>> next_output, _ = model(next_x, hx=(h_n, c_n), return_states=True)
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

    def on_test_epoch_start(self) -> None:
        """Reset the test hidden states"""
        self._test_hidden = [None]

        super().on_test_epoch_start()

    def on_predict_epoch_start(self) -> None:
        """Reset the predict hidden states"""
        self._predict_hidden = [None]

        super().on_predict_epoch_start()

    def common_test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        hx: HIDDEN_STATE | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, HIDDEN_STATE]:
        target = batch.get("target")

        hidden: HIDDEN_STATE
        output, hidden = self.forward(
            batch["input"],
            hx=hx,
            return_states=True,
        )

        hidden = ops.detach(hidden)  # type: ignore[assignment]

        loss_dict = {}
        if target is not None:
            loss = self.criterion(output, target)
            loss_dict["loss"] = loss

        return loss_dict, output, hidden

    def training_step(self, batch: TimeSeriesSample, batch_idx: int) -> StepOutput:
        """
        Perform a single training step for the LSTM model.

        This method processes a training batch, computes the model output and loss,
        and handles the extraction of point predictions for quantile models.

        Parameters
        ----------
        batch : TimeSeriesSample
            Training batch containing input sequences and target values.
            Expected keys: "input", "target".
        batch_idx : int
            Index of the current batch within the training epoch.

        Returns
        -------
        StepOutput
            Dictionary containing:
            - "loss": The computed training loss
            - "output": Model output (detached)
            - "hidden": Final hidden states (detached)
            - "point_prediction": Point estimate for metrics calculation

        Notes
        -----
        The training step:
        1. Processes input through the LSTM model
        2. Computes loss using the specified criterion
        3. Logs training metrics via `common_log_step`
        4. Returns outputs with proper detachment for memory efficiency
        5. Extracts point predictions for quantile models (median quantile)

        Hidden states are detached to prevent gradient accumulation across
        training steps, as each training batch is treated independently.

        Examples
        --------
        >>> # Training step is called automatically by Lightning
        >>> # Manual usage for understanding:
        >>> batch = {"input": torch.randn(32, 100, 5), "target": torch.randn(32, 100, 1)}
        >>> output = model.training_step(batch, batch_idx=0)
        >>> print(output.keys())  # ['loss', 'output', 'hidden', 'point_prediction']
        """
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

    def test_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        """
        Perform a single test step for the LSTM model.

        This method processes a test batch, computes the model output and loss,
        maintaining hidden states across batches within the same test epoch.

        Parameters
        ----------
        batch : TimeSeriesSample
            Test batch containing input sequences and target values.
            Expected keys: "input", "target".
        batch_idx : int
            Index of the current batch within the test epoch.
        dataloader_idx : int, default=0
            Index of the dataloader (for multiple test dataloaders).

        Returns
        -------
        StepOutput
            Dictionary containing:
            - "loss": The computed test loss
            - "output": Model output (detached)
            - "state": Final hidden states (detached)
            - "point_prediction": Point estimate for metrics calculation
        """
        if dataloader_idx >= len(self._test_hidden):
            self._test_hidden.append(None)

        prev_hidden = self._test_hidden[dataloader_idx]

        loss_dict, model_output, hidden = self.common_test_step(
            batch, batch_idx, prev_hidden
        )

        self._test_hidden[dataloader_idx] = hidden

        if loss_dict:
            self.common_log_step(loss_dict, "test")

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

        return typing.cast(StepOutput, loss_dict | output_dict)

    def predict_step(
        self,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single prediction step for the LSTM model.

        This method processes a prediction batch and returns model outputs
        without requiring targets in the batch, maintaining hidden states across batches.

        Parameters
        ----------
        batch : TimeSeriesSample
            Prediction batch containing input sequences.
            Expected keys: "input". "target" is optional.
        batch_idx : int
            Index of the current batch within the prediction.
        dataloader_idx : int, default=0
            Index of the dataloader (for multiple prediction dataloaders).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "output": Model output (detached)
            - "point_prediction": Point predictions for metrics
            - "state": Final hidden states (detached)
        """
        if dataloader_idx >= len(self._predict_hidden):
            self._predict_hidden.append(None)

        prev_hidden = self._predict_hidden[dataloader_idx]

        _, model_output, hidden = self.common_test_step(batch, batch_idx, prev_hidden)

        self._predict_hidden[dataloader_idx] = hidden

        output = {
            "output": ops.detach(model_output),
            "point_prediction": ops.detach(model_output),
            "state": ops.detach(hidden),
        }

        if isinstance(self.criterion, QuantileLoss):
            output["point_prediction"] = self.criterion.point_prediction(model_output)

        return output
