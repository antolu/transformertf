from __future__ import annotations

import collections.abc
import typing

import torch

from ...nn import QuantileLoss
from ...nn._get_activation import VALID_ACTIVATIONS
from .._base_module import (
    DEFAULT_LOGGING_METRICS,
    MetricLiteral,
    setup_criterion_and_output_dim,
)
from .._base_transformer import TransformerModuleBase
from ._model import TemporalConvTransformerModel

__all__ = ["TemporalConvTransformer"]


class TemporalConvTransformer(TransformerModuleBase):
    """
    Lightning module for Temporal Convolutional Transformer (TCT) models.

    This Lightning module wraps the TemporalConvTransformerModel for efficient
    training of long sequence time series forecasting tasks. It combines temporal
    convolutions for sequence compression with transformer attention for capturing
    long-range dependencies.

    The model is particularly suited for:
    - Very long time series sequences (hundreds to thousands of time steps)
    - High-frequency sampled data (1kHz+ as typical in physics applications)
    - Cases where standard transformers are computationally prohibitive
    - Transfer functions and physics-based time series modeling

    Parameters
    ----------
    num_past_features : int
        Number of features in encoder (past) sequences
    num_future_features : int
        Number of features in decoder (future) sequences
    output_dim : int, default=1
        Output dimension for predictions
    hidden_dim : int, default=256
        Hidden dimension for convolution and attention layers
    num_attention_heads : int, default=8
        Number of attention heads in the transformer layer
    compression_factor : int, default=4
        Factor by which to compress sequences before attention.
        Higher values enable longer sequences but may lose information
    num_encoder_layers : int, default=4
        Number of temporal convolution layers in encoder
    num_decoder_layers : int, default=4
        Number of temporal convolution layers in decoder
    dropout : float, default=0.1
        Dropout probability
    activation : VALID_ACTIVATIONS, default="relu"
        Activation function to use throughout the model
    max_dilation : int, default=8
        Maximum dilation factor for temporal convolutions
    causal_attention : bool, default=True
        Whether to use causal attention (each position can only attend to previous positions)
    criterion : torch.nn.Module or None, default=None
        Loss function for training. If None, automatically selects QuantileLoss for
        output_dim > 1 or MSELoss for output_dim = 1
    compile_model : bool, default=False
        Whether to compile the model using PyTorch 2.0 for improved performance
    log_grad_norm : bool, default=False
        Whether to log gradient norms during training for debugging
    trainable_parameters : list[str] or None, default=None
        List of parameter names to train (for transfer learning). If None, train all parameters
    prediction_type : {"point", "delta"}, default="point"
        Type of prediction: "point" for direct values, "delta" for differences
    logging_metrics : Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log during training/validation.
        Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE"

    Attributes
    ----------
    model : TemporalConvTransformerModel
        The core TCT model
    criterion : torch.nn.Module
        Loss function (automatically set to QuantileLoss if output_dim > 1)

    Notes
    -----
    **SEQUENCE LENGTH REQUIREMENTS**:

    This model performs internal sequence compression, requiring minimum sequence
    lengths to function effectively. Using sequences that are too short will
    result in poor performance and potential information loss.

    Recommended minimum lengths:
    - Encoder sequences: >= compression_factor * max_dilation * 12
    - Decoder sequences: >= compression_factor * 8

    For default parameters (compression_factor=4, max_dilation=8):
    - Minimum encoder length: ~384 time steps
    - Minimum decoder length: ~32 time steps

    Ensure your EncoderDecoderDataModule uses sufficient sequence lengths!

    **DATA CHARACTERISTICS**:

    This model is designed for physics applications with:
    - Continuous variables (no categorical features)
    - High sampling rates (1kHz+)
    - No seasonality or periodicity assumptions
    - Focus on transfer functions and temporal dynamics

    **PERFORMANCE SCALING**:

    The computational complexity scales as O(n/c²) where n is sequence length
    and c is compression_factor, making it much more efficient than standard
    transformers for long sequences.

    Examples
    --------
    >>> from transformertf.models.temporal_conv_transformer import TemporalConvTransformer
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> import lightning as L
    >>>
    >>> # Basic TCT model for long sequence forecasting
    >>> model = TemporalConvTransformer(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     output_dim=1,
    ...     hidden_dim=256,
    ...     compression_factor=4
    ... )
    >>>
    >>> # Data module with long sequences
    >>> data_module = EncoderDecoderDataModule(
    ...     train_df_paths=["long_sequences.parquet"],
    ...     target_covariate="target",
    ...     ctxt_seq_len=500,  # Long context
    ...     tgt_seq_len=100,   # Prediction horizon
    ... )
    >>>
    >>> # Train with sequence length validation
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(model, data_module)

    >>> # High compression for very long sequences
    >>> model = TemporalConvTransformer(
    ...     num_past_features=20,
    ...     num_future_features=10,
    ...     compression_factor=8,  # More aggressive compression
    ...     hidden_dim=512,
    ...     max_dilation=16
    ... )

    >>> # Using TCT alias for convenience
    >>> from transformertf.models.temporal_conv_transformer import TCT
    >>> model = TCT(num_past_features=15, num_future_features=8)

    See Also
    --------
    TemporalConvTransformerModel : Core model implementation
    TransformerModuleBase : Base class with training/validation logic
    EncoderDecoderDataModule : Compatible data module for encoder-decoder tasks
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        compression_factor: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        max_dilation: int = 8,
        causal_attention: bool = True,
        criterion: torch.nn.Module | None = None,
        compile_model: bool = False,
        log_grad_norm: bool = False,
        trainable_parameters: list[str] | None = None,
        prediction_type: typing.Literal["point", "delta"] = "point",
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # Set up criterion and adjust output_dim using shared logic
        self.criterion, output_dim = setup_criterion_and_output_dim(
            criterion, output_dim
        )
        if isinstance(self.criterion, QuantileLoss):
            self.hparams["output_dim"] = output_dim

        # Create the core model with correct output_dim
        self.model = TemporalConvTransformerModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            compression_factor=compression_factor,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            max_dilation=max_dilation,
            causal_attention=causal_attention,
        )

    def forward(
        self,
        batch=None,
        encoder_input=None,
        decoder_input=None,
        encoder_lengths=None,
        decoder_lengths=None,
    ):
        """Forward pass through the TCT model."""
        # Support both batch dict and individual arguments
        if batch is not None:
            return self.model(
                encoder_input=batch["encoder_input"],
                decoder_input=batch["decoder_input"],
                encoder_lengths=batch.get("encoder_lengths"),
                decoder_lengths=batch.get("decoder_lengths"),
            )
        return self.model(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
        )
