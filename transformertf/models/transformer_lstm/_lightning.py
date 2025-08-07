from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import MSELoss, QuantileLoss
from .._base_module import DEFAULT_LOGGING_METRICS, MetricLiteral
from .._base_transformer import TransformerModuleBase
from .._validation_mixin import EncoderAlignmentValidationMixin
from ._model import HIDDEN_STATE, TransformerLSTMModel

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="TransformerLSTM")


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    encoder_states: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor]


class TransformerLSTM(TransformerModuleBase, EncoderAlignmentValidationMixin):
    """
    Lightning module for TransformerLSTMModel for sequence-to-sequence forecasting.

    This Lightning module wraps the TransformerLSTMModel and provides the training,
    validation, and testing interfaces required by PyTorch Lightning. It combines
    LSTM encoder-decoder architecture with multiple transformer blocks featuring
    self-attention and cross-attention mechanisms.

    The module supports both point predictions and probabilistic forecasting via
    quantile regression, with enhanced capabilities through transformer attention.

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
    d_model : int, default=128
        Model dimension used for LSTM layers and transformer blocks.
    num_layers : int, default=2
        Number of LSTM layers for both encoder and decoder.
    num_transformer_blocks : int, default=2
        Number of transformer blocks to apply after LSTM processing.
    dropout : float, default=0.1
        Dropout probability applied to all components.
    num_heads : int, default=4
        Number of attention heads in transformer blocks.
    output_dim : int, default=1
        Output dimension. For quantile loss, this is automatically set to the number of quantiles.
    causal_attention : bool, default=True
        Whether to use causal attention masking.
    criterion : torch.nn.Module | None, default=None
        Loss function for training. If None, defaults to MSELoss.
        Use QuantileLoss for probabilistic forecasting.
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
    model : TransformerLSTMModel
        The underlying transformer-enhanced LSTM model.
    criterion : torch.nn.Module
        The loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models.transformer_lstm import TransformerLSTM
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create model for point prediction
    >>> model = TransformerLSTM(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     d_model=64,
    ...     num_layers=2,
    ...     num_transformer_blocks=3,
    ...     num_heads=4,
    ...     output_dim=1
    ... )
    >>>
    >>> # Create model for probabilistic forecasting
    >>> quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    >>> model = TransformerLSTM(
    ...     num_past_features=8,
    ...     num_future_features=3,
    ...     d_model=128,
    ...     num_transformer_blocks=4,
    ...     num_heads=8,
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
    - `encoder_lengths`: Actual lengths of encoder sequences (B, 1)
    - `decoder_lengths`: Actual lengths of decoder sequences (B, 1)
    - `target`: Target values (B, future_seq_len, output_dim) for training/validation

    **Transformer Enhancement:**

    Unlike basic LSTM models, this model includes:
    - Multiple transformer blocks with self-attention and cross-attention
    - Proper attention masking for variable sequence lengths
    - GLU-based residual connections in transformer blocks
    - Enhanced sequence modeling capabilities

    **Architecture Flow:**

    1. LSTM encoder processes past sequence
    2. LSTM decoder processes future sequence with encoder context
    3. N transformer blocks apply self-attention and cross-attention
    4. Final linear layer produces predictions

    **Quantile Support:**

    When using QuantileLoss, the model automatically:
    - Sets output_dim to the number of quantiles
    - Extracts point predictions (median) from quantile outputs
    - Handles probabilistic forecasting workflow

    See Also
    --------
    TransformerLSTMModel : The underlying model implementation
    TransformerModuleBase : Base class for transformer-based models
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_transformer_blocks: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        output_dim: int = 1,
        causal_attention: bool = True,
        share_lstm_weights: bool = False,
        criterion: torch.nn.Module | None = None,
        *,
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
        self.criterion = criterion or MSELoss()

        # Handle quantile loss case
        if isinstance(self.criterion, QuantileLoss):
            output_dim = len(self.criterion.quantiles)
            self.hparams["output_dim"] = output_dim

        # Create the model
        self.model = TransformerLSTMModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            d_model=d_model,
            num_layers=num_layers,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout,
            num_heads=num_heads,
            output_dim=output_dim,
            causal_attention=causal_attention,
            share_lstm_weights=share_lstm_weights,
        )

    def on_train_batch_start(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> None:
        """Validate encoder alignment on first training batch."""
        if batch_idx == 0 and self.current_epoch == 0:
            self.validate_encoder_alignment_with_batch(batch)
        super().on_train_batch_start(batch, batch_idx)

    def forward(
        self,
        batch: EncoderDecoderTargetSample,
    ) -> dict[str, torch.Tensor | HIDDEN_STATE]:
        """
        Forward pass through the model.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Batch containing encoder_input, decoder_input, encoder_lengths,
            decoder_lengths, and optionally target.

        Returns
        -------
        dict[str, torch.Tensor | HIDDEN_STATE]
            Dictionary containing:
            - "output": Model output tensor
            - "encoder_states": Encoder hidden states
        """
        # Extract and reshape lengths like PF-TFT and AttentionLSTM
        encoder_lengths = batch.get("encoder_lengths")
        decoder_lengths = batch.get("decoder_lengths")
        if encoder_lengths is not None:
            encoder_lengths = encoder_lengths[..., 0]  # (B, 1) -> (B,)
        if decoder_lengths is not None:
            decoder_lengths = decoder_lengths[..., 0]  # (B, 1) -> (B,)

        # Slice decoder inputs to keep only num_future_features (like PF-TFT and AttentionLSTM)
        decoder_inputs = batch["decoder_input"][
            ..., : self.hparams["num_future_features"]
        ]

        output, states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=decoder_inputs,
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            return_encoder_states=True,
        )

        return {
            "output": output,
            "encoder_states": states,
        }
