from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from ...utils import ops
from .._base_module import DEFAULT_LOGGING_METRICS, LightningModuleBase, MetricLiteral
from ._model import HIDDEN_STATE, AttentionLSTM

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="AttentionLSTMModule")


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    encoder_states: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor]


class AttentionLSTMModule(LightningModuleBase):
    """
    Lightning module for AttentionLSTM model for sequence-to-sequence forecasting.

    This Lightning module wraps the AttentionLSTM model and provides the training,
    validation, and testing interfaces required by PyTorch Lightning. It supports both point
    predictions and probabilistic forecasting via quantile regression, with enhanced capabilities
    through self-attention mechanisms.

    The module is designed to work with EncoderDecoderTargetSample data format with dynamic
    sequence lengths, utilizing encoder_lengths and decoder_lengths for proper masking.

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
    hidden_size : int, default=128
        Hidden size used for both encoder and decoder LSTM layers.
    num_layers : int, default=2
        Number of LSTM layers used for both encoder and decoder.
    dropout : float, default=0.1
        Dropout probability applied to all components (LSTM, attention, output).
    n_heads : int, default=4
        Number of attention heads in the multi-head attention mechanism.
    use_gating : bool, default=True
        Whether to use gating mechanism for skip connections.
    trainable_add : bool, default=False
        Whether to use learnable gating for the residual connection.
    output_dim : int, default=1
        Output dimension. For quantile loss, this is automatically set to the number of quantiles.
    criterion : torch.nn.Module | None, default=None
        Loss function for training. If None, defaults to MSELoss.
        Use QuantileLoss for probabilistic forecasting.
    prediction_type : {"delta", "point"}, default="point"
        Type of prediction target:
        - "point": Predict absolute values directly
        - "delta": Predict differences between consecutive time steps
    log_grad_norm : bool, default=False
        Whether to log gradient norms during training for debugging.
    compile_model : bool, default=False
        Whether to compile the model using PyTorch 2.0 for improved performance.
    trainable_parameters : list[str] | None, default=None
        List of parameter names to train. If None, all parameters are trainable.
        Useful for transfer learning scenarios.
    logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log. If empty, only loss is logged.

    Attributes
    ----------
    model : AttentionLSTM
        The underlying attention-enhanced LSTM model.
    criterion : torch.nn.Module
        The loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models.attention_lstm import AttentionLSTMModule
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create model for point prediction
    >>> model = AttentionLSTMModule(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     n_heads=4,
    ...     use_gating=True,
    ...     output_dim=1
    ... )
    >>>
    >>> # Create model for probabilistic forecasting
    >>> quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    >>> model = AttentionLSTMModule(
    ...     num_past_features=8,
    ...     num_future_features=3,
    ...     hidden_size=128,
    ...     n_heads=8,
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
    - `decoder_lengths`: Actual lengths of decoder sequences (B,)
    - `target`: Target values (B, future_seq_len, output_dim) for training/validation

    **Dynamic Length Support:**

    The model uses decoder_lengths to create attention masks for variable sequence
    lengths, ensuring proper handling of padded sequences in batches.

    **Self-Attention Enhancement:**

    Unlike the basic encoder-decoder LSTM, this model includes self-attention on
    decoder outputs with optional gating for enhanced sequence modeling capabilities.

    **Quantile Support:**

    When using QuantileLoss, the model automatically:
    - Sets output_dim to the number of quantiles
    - Extracts point predictions (median) from quantile outputs
    - Handles probabilistic forecasting workflow

    See Also
    --------
    AttentionLSTM : The underlying model implementation
    LightningModuleBase : Base class for all Lightning modules
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
        use_gating: bool = True,
        trainable_add: bool = False,
        output_dim: int = 1,
        criterion: torch.nn.Module | None = None,
        *,
        prediction_type: typing.Literal["delta", "point"] = "point",
        log_grad_norm: bool = False,
        compile_model: bool = False,
        trainable_parameters: list[str] | None = None,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # Set up criterion
        self.criterion = criterion or torch.nn.MSELoss()

        # Handle quantile loss case
        if isinstance(self.criterion, QuantileLoss):
            output_dim = len(self.criterion.quantiles)
            self.hparams["output_dim"] = output_dim

        # Create the model
        self.model = AttentionLSTM(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_heads=n_heads,
            use_gating=use_gating,
            trainable_add=trainable_add,
            output_dim=output_dim,
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
            Batch containing encoder_input, decoder_input, decoder_lengths, and optionally target.
        return_encoder_states : bool, default=False
            Whether to return encoder hidden states along with output.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]
            Model output, optionally with encoder states.
        """
        decoder_lengths = batch.get("decoder_lengths")
        if decoder_lengths is not None and decoder_lengths.dim() > 1:
            # Remove the last dimension if it's (B, 1) -> (B,)
            decoder_lengths = decoder_lengths.squeeze(-1)

        return self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            decoder_lengths=decoder_lengths,
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

    def _common_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int, stage: str
    ) -> StepOutput:
        """Common logic for validation and test steps."""
        loss, model_output = self._compute_loss_and_output(batch)

        # Get encoder states
        _, encoder_states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"],
            decoder_lengths=batch.get("decoder_lengths"),
            return_encoder_states=True,
        )

        # Log metrics
        self.common_log_step({"loss": loss}, stage)

        # Prepare output dictionary
        output_dict = {
            "loss": loss,
            "output": ops.detach(model_output),
            "encoder_states": ops.detach(encoder_states),
        }

        # Extract point prediction for metrics
        output_dict["point_prediction"] = self._extract_point_prediction(model_output)

        return typing.cast(StepOutput, output_dict)

    def _extract_point_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        """Extract point prediction from model output for quantile models."""
        if isinstance(self.criterion, QuantileLoss):
            return self.criterion.point_prediction(model_output)
        return model_output

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> StepOutput:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Training batch containing encoder_input, decoder_input, decoder_lengths, and target.
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
            decoder_lengths=batch.get("decoder_lengths"),
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
        output_dict["point_prediction"] = self._extract_point_prediction(model_output)

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
            Validation batch containing encoder_input, decoder_input, decoder_lengths, and target.
        batch_idx : int
            Index of the current batch within the validation epoch.
        dataloader_idx : int, default=0
            Index of the validation dataloader (for multiple validation sets).

        Returns
        -------
        StepOutput
            Dictionary containing loss, output, encoder_states, and point_prediction.
        """
        return self._common_step(batch, batch_idx, "validation")

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
            Test batch containing encoder_input, decoder_input, decoder_lengths, and target.
        batch_idx : int
            Index of the current batch within the test epoch.
        dataloader_idx : int, default=0
            Index of the test dataloader (for multiple test sets).

        Returns
        -------
        StepOutput
            Dictionary containing loss, output, encoder_states, and point_prediction.
        """
        return self._common_step(batch, batch_idx, "test")

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
            Prediction batch containing encoder_input, decoder_input, and decoder_lengths.
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
            decoder_lengths=batch.get("decoder_lengths"),
            return_encoder_states=True,
        )

        return {
            "output": output,
            "encoder_states": encoder_states,
            "point_prediction": self._extract_point_prediction(output),
        }
