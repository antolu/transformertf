from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import VALID_ACTIVATIONS, QuantileLoss
from .._base_module import DEFAULT_LOGGING_METRICS, MetricLiteral
from .._base_transformer import TransformerModuleBase
from .._validation_mixin import EncoderAlignmentValidationMixin
from ._model import HIDDEN_STATE, EncoderDecoderLSTMModel

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="EncoderDecoderLSTM")


class EncoderDecoderLSTM(TransformerModuleBase, EncoderAlignmentValidationMixin):
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
    compile_model : bool, default=False
        Whether to compile the model using torch.compile for performance optimization.
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
        compile_model: bool = False,
        trainable_parameters: list[str] | dict[str, list[str]] | None = None,
        *,
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
        self.model = EncoderDecoderLSTMModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            d_mlp_hidden=d_mlp_hidden,
            output_dim=output_dim,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
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
            Batch containing encoder_input, decoder_input, and optionally target.

        Returns
        -------
        dict[str, torch.Tensor | HIDDEN_STATE]
            Dictionary containing 'output' tensor and 'encoder_states' tuple.
        """
        encoder_lengths = batch.get("encoder_lengths")
        decoder_lengths = batch.get("decoder_lengths")
        if encoder_lengths is not None:
            encoder_lengths = encoder_lengths[..., 0]
        if decoder_lengths is not None:
            decoder_lengths = decoder_lengths[..., 0]

        output, encoder_states = self.model(
            past_sequence=batch["encoder_input"],
            future_sequence=batch["decoder_input"][
                ..., : self.hparams["num_future_features"]
            ],
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
            return_encoder_states=True,
        )

        return {
            "output": output,
            "encoder_states": encoder_states,
        }
