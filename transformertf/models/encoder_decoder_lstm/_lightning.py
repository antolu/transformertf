from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import VALID_ACTIVATIONS, QuantileLoss
from ...utils import ops
from .._base_module import DEFAULT_LOGGING_METRICS, LightningModuleBase, MetricLiteral
from ._model import HIDDEN_STATE, EncoderDecoderLSTMModel

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="EncoderDecoderLSTM")


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    encoder_states: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor]


class EncoderDecoderLSTM(LightningModuleBase):
    """
    Lightning module for EncoderDecoderLSTM model for sequence-to-sequence forecasting.

    This Lightning module wraps the EncoderDecoderLSTMModel and provides the training,
    validation, and testing interfaces required by PyTorch Lightning. It supports both
    point predictions and probabilistic forecasting via quantile regression.

    The module is designed to work with EncoderDecoderTargetSample data format, which
    provides separate encoder input (past features) and decoder input (future features)
    along with target values for supervised learning.

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
    d_encoder : int, default=128
        Hidden size of the encoder LSTM layers.
    d_decoder : int, default=128
        Hidden size of the decoder LSTM layers.
    num_encoder_layers : int, default=2
        Number of LSTM layers in the encoder.
    num_decoder_layers : int, default=2
        Number of LSTM layers in the decoder.
    dropout : float, default=0.1
        Dropout probability for LSTM layers.
    d_mlp_hidden : int | tuple[int, ...] | None, default=None
        Hidden dimensions for the MLP head. If None, uses a single linear layer.
    output_dim : int, default=1
        Output dimension. For quantile loss, this is automatically set to the number of quantiles.
    mlp_activation : VALID_ACTIVATIONS, default="relu"
        Activation function for the MLP head.
    mlp_dropout : float, default=0.1
        Dropout probability for the MLP head.
    criterion : torch.nn.Module | None, default=None
        Loss function for training. If None, defaults to MSELoss.
        Use QuantileLoss for probabilistic forecasting.
    logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log. If empty, only loss is logged.

    Attributes
    ----------
    model : EncoderDecoderLSTMModel
        The underlying encoder-decoder LSTM model.
    criterion : torch.nn.Module
        The loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models.encoder_decoder_lstm import EncoderDecoderLSTM
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create model for point prediction
    >>> model = EncoderDecoderLSTM(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     encoder_d_model=64,
    ...     decoder_d_model=64,
    ...     d_mlp_hidden=(32, 16),
    ...     output_dim=1
    ... )
    >>>
    >>> # Create model for probabilistic forecasting
    >>> quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    >>> model = EncoderDecoderLSTM(
    ...     num_past_features=8,
    ...     num_future_features=3,
    ...     encoder_d_model=128,
    ...     decoder_d_model=128,
    ...     criterion=QuantileLoss(quantiles=quantiles)
    ... )
    >>>
    >>> # Train with encoder-decoder data
    >>> datamodule = EncoderDecoderDataModule(...)
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)

    Notes
    -----
    **Data Format:**

    The module expects EncoderDecoderTargetSample with:
    - `encoder_input`: Past features (B, past_seq_len, num_past_features)
    - `decoder_input`: Future features (B, future_seq_len, num_future_features)
    - `target`: Target values (B, future_seq_len, output_dim) for training/validation

    **Hidden State Management:**

    Unlike the simple LSTM model, this module doesn't maintain hidden states
    across validation batches since the encoder-decoder architecture processes
    complete sequences independently.

    **Teacher Forcing:**

    During training, the model uses teacher forcing where the true future features
    are provided to the decoder. During inference, the model can operate with
    provided future features or use autoregressive generation.

    **Quantile Support:**

    When using QuantileLoss, the model automatically:
    - Sets output_dim to the number of quantiles
    - Extracts point predictions (median) from quantile outputs
    - Handles probabilistic forecasting workflow

    See Also
    --------
    EncoderDecoderLSTMModel : The underlying model implementation
    LightningModuleBase : Base class for all Lightning modules
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        d_encoder: int = 128,
        d_decoder: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        d_mlp_hidden: int | tuple[int, ...] | None = None,
        output_dim: int = 1,
        mlp_activation: VALID_ACTIVATIONS = "relu",
        mlp_dropout: float = 0.1,
        criterion: torch.nn.Module | None = None,
        *,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # Store metrics configuration
        self._logging_metrics = getattr(
            self.hparams, "logging_metrics", logging_metrics
        )

        # Set up criterion
        self.criterion = criterion or torch.nn.MSELoss()

        # Handle quantile loss case
        if isinstance(self.criterion, QuantileLoss):
            output_dim = len(self.criterion.quantiles)
            self.hparams["output_dim"] = output_dim

        # Create the model
        self.model = EncoderDecoderLSTMModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            encoder_d_model=d_encoder,
            decoder_d_model=d_decoder,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            d_mlp_hidden=d_mlp_hidden,
            output_dim=output_dim,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
        )

    def forward(
        self,
        batch: EncoderDecoderTargetSample,
        *,
        return_encoder_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]:
        """
        Forward pass through the model.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Batch containing encoder_input, decoder_input, and optionally target.
        return_encoder_states : bool, default=False
            Whether to return encoder hidden states along with output.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]
            Model output, optionally with encoder states.
        """
        return self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            return_encoder_states=return_encoder_states,
        )

    def _compute_loss_and_output(
        self, batch: EncoderDecoderTargetSample
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute model output and loss for a batch."""
        assert "target" in batch, "Target is required for loss computation"

        output = self(batch)
        loss = self.criterion(output, batch["target"])

        return loss, output

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> StepOutput:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Training batch containing encoder_input, decoder_input, and target.
        batch_idx : int
            Index of the current batch within the training epoch.

        Returns
        -------
        StepOutput
            Dictionary containing loss, output, encoder_states, and point_prediction.
        """
        loss, model_output = self._compute_loss_and_output(batch)

        # Get encoder states for potential analysis
        _, encoder_states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            return_encoder_states=True,
        )

        # Log training metrics
        self.common_log_step({"loss": loss}, "train")

        # Prepare output dictionary
        output_dict = {
            "loss": loss,
            "output": ops.detach(model_output),
            "encoder_states": ops.detach(encoder_states),
        }

        # Extract point prediction for metrics
        if isinstance(self.criterion, QuantileLoss):
            output_dict["point_prediction"] = self.criterion.point_prediction(
                model_output
            )
        else:
            output_dict["point_prediction"] = output_dict["output"]

        return typing.cast(StepOutput, output_dict)

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Validation batch containing encoder_input, decoder_input, and target.
        batch_idx : int
            Index of the current batch within the validation epoch.
        dataloader_idx : int, default=0
            Index of the validation dataloader (for multiple validation sets).

        Returns
        -------
        StepOutput
            Dictionary containing loss, output, encoder_states, and point_prediction.
        """
        loss, model_output = self._compute_loss_and_output(batch)

        # Get encoder states
        _, encoder_states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            return_encoder_states=True,
        )

        # Log validation metrics
        self.common_log_step({"loss": loss}, "validation")

        # Prepare output dictionary
        output_dict = {
            "loss": loss,
            "output": ops.detach(model_output),
            "encoder_states": ops.detach(encoder_states),
        }

        # Extract point prediction for metrics
        if isinstance(self.criterion, QuantileLoss):
            output_dict["point_prediction"] = self.criterion.point_prediction(
                model_output
            )
        else:
            output_dict["point_prediction"] = output_dict["output"]

        return typing.cast(StepOutput, output_dict)

    def test_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        """
        Perform a single test step.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Test batch containing encoder_input, decoder_input, and target.
        batch_idx : int
            Index of the current batch within the test epoch.
        dataloader_idx : int, default=0
            Index of the test dataloader (for multiple test sets).

        Returns
        -------
        StepOutput
            Dictionary containing loss, output, encoder_states, and point_prediction.
        """
        loss, model_output = self._compute_loss_and_output(batch)

        # Get encoder states
        _, encoder_states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            return_encoder_states=True,
        )

        # Log test metrics
        self.common_log_step({"loss": loss}, "test")

        # Prepare output dictionary
        output_dict = {
            "loss": loss,
            "output": ops.detach(model_output),
            "encoder_states": ops.detach(encoder_states),
        }

        # Extract point prediction for metrics
        if isinstance(self.criterion, QuantileLoss):
            output_dict["point_prediction"] = self.criterion.point_prediction(
                model_output
            )
        else:
            output_dict["point_prediction"] = output_dict["output"]

        return typing.cast(StepOutput, output_dict)

    def predict_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single prediction step.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Prediction batch containing encoder_input and decoder_input.
            Target is optional for prediction.
        batch_idx : int
            Index of the current batch within the prediction phase.
        dataloader_idx : int, default=0
            Index of the prediction dataloader.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing model output and encoder states.
        """
        # Get model output and encoder states
        output, encoder_states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            return_encoder_states=True,
        )

        result = {
            "output": output,
            "encoder_states": encoder_states,
        }

        # Add point prediction for quantile models
        if isinstance(self.criterion, QuantileLoss):
            result["point_prediction"] = self.criterion.point_prediction(output)
        else:
            result["point_prediction"] = output

        return result
