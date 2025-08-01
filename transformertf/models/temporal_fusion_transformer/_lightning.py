from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_module import (
    DEFAULT_LOGGING_METRICS,
    MetricLiteral,
    setup_criterion_and_output_dim,
)
from .._base_transformer import TransformerModuleBase
from ._model import TemporalFusionTransformerModel


class TemporalFusionTransformer(TransformerModuleBase):
    """
    Temporal Fusion Transformer (TFT) for multivariate time series forecasting.

    This model implements the Temporal Fusion Transformer architecture as described in
    "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    by Lim et al. (2021). TFT combines the strengths of LSTMs and self-attention mechanisms
    to handle complex temporal patterns in multivariate time series.

    The model is designed for encoder-decoder forecasting tasks where historical context
    (past covariates) and future known information (future covariates) are available.
    It provides both point predictions and probabilistic forecasts via quantile regression.

    Parameters
    ----------
    num_past_features : int
        Number of features in the historical context sequence (encoder input).
        These are the time-varying covariates observed in the past.
    num_future_features : int
        Number of features in the future known information (decoder input).
        These are the time-varying covariates known for the prediction horizon.
    ctxt_seq_len : int
        Length of the context (encoder) sequence. This determines how much
        historical information the model can access.
    tgt_seq_len : int
        Length of the target (decoder) sequence. This is the prediction horizon.
    d_model : int, default=300
        Dimension of the model's hidden representations. This affects model
        capacity and computational requirements.
    hidden_continuous_dim : int, default=8
        Dimension of the continuous variable embeddings in the variable
        selection networks.
    num_heads : int, default=4
        Number of attention heads in the multi-head attention mechanism.
    num_lstm_layers : int, default=2
        Number of LSTM layers in both encoder and decoder components.
    dropout : float, default=0.1
        Dropout probability applied throughout the model for regularization.
    output_dim : int, default=7
        Output dimension, typically the number of quantiles for probabilistic
        forecasting. Automatically set when using QuantileLoss.
    criterion : QuantileLoss or torch.nn.Module or None, default=None
        Loss function for training. If None, defaults to QuantileLoss().
        For probabilistic forecasting, use QuantileLoss with desired quantiles.
    casual_attention : bool, default=True
        Whether to use causal attention masking in the decoder. Should be True
        for autoregressive forecasting.
    prediction_type : {"delta", "point"}, default="point"
        Type of prediction target:
        - "point": Predict absolute values directly
        - "delta": Predict differences between consecutive time steps
    log_grad_norm : bool, default=False
        Whether to log gradient norms during training for debugging.
    compile_model : bool, default=False
        Whether to compile the model using PyTorch 2.0 for improved performance.
    trainable_parameters : list[str] or None, default=None
        List of parameter names to train. If None, all parameters are trainable.
        Useful for transfer learning scenarios.
    logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log during training, validation, and testing.
        If empty, no additional metrics will be logged (only the loss from the criterion).
        Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE".

    Attributes
    ----------
    model : TemporalFusionTransformerModel
        The underlying TFT model implementation.
    criterion : torch.nn.Module
        The loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models import TemporalFusionTransformer
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create model with quantile loss for probabilistic forecasting
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = TemporalFusionTransformer(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     ctxt_seq_len=168,  # 1 week of hourly data
    ...     tgt_seq_len=24,    # 1 day prediction horizon
    ...     d_model=256,
    ...     criterion=QuantileLoss(quantiles=quantiles),
    ...     compile_model=True
    ... )
    >>>
    >>> # Train with data module
    >>> datamodule = EncoderDecoderDataModule(...)
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
    >>>
    >>> # Generate predictions
    >>> predictions = trainer.predict(model, datamodule.test_dataloader())

    Notes
    -----
    **Architecture Overview:**

    The TFT model consists of several key components:

    1. **Variable Selection Networks**: Learn to select relevant features
       from past and future covariates dynamically.

    2. **LSTM Encoder-Decoder**: Processes sequential information and
       captures temporal dependencies.

    3. **Multi-Head Attention**: Captures long-range dependencies and
       provides interpretability through attention weights.

    4. **Gated Residual Networks**: Enable deep residual learning with
       gating mechanisms for gradient flow.

    **Input Requirements:**

    The model expects :class:`transformertf.data.EncoderDecoderTargetSample` with:
    - `encoder_input`: Past covariates (B, ctxt_seq_len, num_past_features)
    - `decoder_input`: Future covariates (B, tgt_seq_len, num_future_features)
    - `encoder_lengths`: Actual lengths of encoder sequences
    - `decoder_lengths`: Actual lengths of decoder sequences

    **Static Covariates:**

    The model automatically generates static covariates based on encoder lengths,
    which can capture sequence-level information that affects the entire forecast.

    References
    ----------
    .. [1] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
           multi-horizon time series forecasting." International Conference on
           Machine Learning. PMLR, 2021.

    See Also
    --------
    TransformerModuleBase : Base class for transformer models
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    transformertf.nn.VariableSelection : Variable selection networks
    transformertf.nn.InterpretableMultiHeadAttention : Interpretable attention mechanism
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        *,
        casual_attention: bool = True,
        prediction_type: typing.Literal["delta", "point"] | None = None,
        log_grad_norm: bool = False,
        compile_model: bool = False,
        trainable_parameters: list[str] | None = None,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # Set up criterion and adjust output_dim using shared logic
        self.criterion, output_dim = setup_criterion_and_output_dim(
            criterion,
            output_dim,
            default_quantiles=None,  # Use QuantileLoss default
        )
        if isinstance(self.criterion, QuantileLoss):
            self.hparams["output_dim"] = output_dim

        self.model = TemporalFusionTransformerModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            d_model=d_model,
            num_static_features=1,
            hidden_continuous_dim=hidden_continuous_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            output_dim=output_dim,
            casual_attention=casual_attention,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> dict[str, torch.Tensor]:
        """
        Forward pass through the Temporal Fusion Transformer model.

        This method processes the input batch through the TFT architecture, handling
        the extraction of past and future covariates, static covariate generation,
        and sequence length information.

        Parameters
        ----------
        x : EncoderDecoderTargetSample
            Input batch containing the following keys:
            - "encoder_input": Past covariates (B, ctxt_seq_len, num_past_features)
            - "decoder_input": Future covariates (B, tgt_seq_len, total_features)
            - "encoder_lengths": Actual lengths of encoder sequences (B, 1)
            - "decoder_lengths": Actual lengths of decoder sequences (B, 1)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing model outputs:
            - "output": Main predictions (B, tgt_seq_len, output_dim)
            - Additional outputs may include attention weights, variable selection
              weights, and other interpretable components depending on the
              underlying model configuration.

        Notes
        -----
        **Static Covariates Generation:**

        The method automatically generates static covariates by normalizing encoder
        lengths to the range [-1, 1]. This provides the model with information about
        the relative sequence completeness:

        .. code-block:: python

            static_covariates = encoder_lengths * 2 / ctxt_seq_len - 1

        **Future Covariates Extraction:**

        The method extracts only the relevant future features from the decoder input,
        which may contain additional features like targets from teacher forcing:

        .. code-block:: python

            future_covariates = decoder_input[..., -num_future_features:]

        **Input Shape Requirements:**

        - encoder_input: (batch_size, ctxt_seq_len, num_past_features)
        - decoder_input: (batch_size, tgt_seq_len, total_features)
        - encoder_lengths: (batch_size, 1)
        - decoder_lengths: (batch_size, 1)

        Examples
        --------
        >>> batch = {
        ...     "encoder_input": torch.randn(32, 168, 10),
        ...     "decoder_input": torch.randn(32, 24, 15),
        ...     "encoder_lengths": torch.full((32, 1), 168),
        ...     "decoder_lengths": torch.full((32, 1), 24)
        ... }
        >>> model = TemporalFusionTransformer(
        ...     num_past_features=10,
        ...     num_future_features=5,
        ...     ctxt_seq_len=168,
        ...     tgt_seq_len=24
        ... )
        >>> output = model(batch)
        >>> print(output["output"].shape)  # torch.Size([32, 24, 7])
        """
        static_covariates = x["encoder_lengths"] * 2 / self.hparams["ctxt_seq_len"] - 1

        return self.model(
            past_covariates=x["encoder_input"],  # (B, T, F_past)
            future_covariates=x["decoder_input"][  # (B, T, F_future)
                ...,
                -self.hparams["num_future_features"] :,
            ],
            static_covariates=static_covariates,  # type: ignore[typeddict-item]
            encoder_lengths=x["encoder_lengths"][..., 0],  # (B, T)
            decoder_lengths=x["decoder_lengths"][..., 0],  # (B, T)
        )
