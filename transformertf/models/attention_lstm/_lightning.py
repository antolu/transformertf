from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_module import DEFAULT_LOGGING_METRICS, MetricLiteral
from .._base_transformer import TransformerModuleBase
from ._model import HIDDEN_STATE, AttentionLSTMModel

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="AttentionLSTM")


class StepOutput(typing.TypedDict):
    loss: torch.Tensor
    output: torch.Tensor
    encoder_states: HIDDEN_STATE
    point_prediction: typing.NotRequired[torch.Tensor]


class AttentionLSTM(TransformerModuleBase):
    """
    Lightning module for AttentionLSTMModel for sequence-to-sequence forecasting.

    This Lightning module wraps the AttentionLSTMModel and provides the training,
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
        causal_attention: bool = True,
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
        self.model = AttentionLSTMModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_heads=n_heads,
            use_gating=use_gating,
            trainable_add=trainable_add,
            output_dim=output_dim,
            causal_attention=causal_attention,
        )

    def forward(
        self,
        batch: EncoderDecoderTargetSample,
    ) -> torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]:
        """
        Forward pass through the model.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Batch containing encoder_input, decoder_input, encoder_lengths, decoder_lengths, and optionally target.
        return_encoder_states : bool, default=False
            Whether to return encoder hidden states along with output.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]
            Model output, optionally with encoder states.
        """
        # Extract and reshape lengths like PF-TFT
        encoder_lengths = batch.get("encoder_lengths")
        decoder_lengths = batch.get("decoder_lengths")
        if encoder_lengths is not None:
            encoder_lengths = encoder_lengths[..., 0]  # (B, 1) -> (B,)
        if decoder_lengths is not None:
            decoder_lengths = decoder_lengths[..., 0]  # (B, 1) -> (B,)

        # Slice decoder inputs to keep only num_future_features (like PF-TFT)
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
