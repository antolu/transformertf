from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_module import DEFAULT_LOGGING_METRICS, MetricLiteral
from .._base_transformer import TransformerModuleBase
from ._model import VanillaTransformerModel


class VanillaTransformer(TransformerModuleBase):
    """
    Vanilla Transformer model for time series forecasting.

    This model implements a standard transformer architecture adapted for time series
    forecasting tasks. It uses the classic encoder-decoder structure with multi-head
    self-attention mechanisms to capture long-range dependencies in temporal data.

    The model follows the original "Attention Is All You Need" design but is specifically
    configured for sequence-to-sequence forecasting where historical context is encoded
    and future predictions are generated autoregressively.

    Parameters
    ----------
    num_features : int
        Number of input features in the time series.
    ctxt_seq_len : int
        Length of the context (encoder) sequence for historical data.
    tgt_seq_len : int
        Length of the target (decoder) sequence for predictions.
    d_model : int, default=128
        Dimension of the model's hidden representations. Must be divisible
        by num_heads for multi-head attention.
    num_heads : int, default=8
        Number of attention heads in multi-head attention mechanisms.
        Should divide d_model evenly.
    num_encoder_layers : int, default=6
        Number of transformer encoder layers. More layers can capture
        more complex patterns but increase computational cost.
    num_decoder_layers : int, default=6
        Number of transformer decoder layers. More layers can improve
        generation quality but increase computational cost.
    dropout : float, default=0.1
        Dropout probability applied throughout the model for regularization.
    activation : str, default="relu"
        Activation function used in feed-forward networks within transformer blocks.
        Common choices: "relu", "gelu".
    fc_dim : int or tuple[int, ...], default=1024
        Dimension(s) of the feed-forward network within transformer blocks.
        If int, creates a single layer. If tuple, creates multi-layer MLP.
    output_dim : int, default=7
        Output dimension. For quantile loss, this is automatically set
        to the number of quantiles.
    criterion : QuantileLoss or None, default=None
        Loss function for training. If None, defaults to QuantileLoss().
        Supports both point and probabilistic forecasting.
    prediction_type : {"delta", "point"}, default="point"
        Type of prediction target:
        - "point": Predict absolute values directly
        - "delta": Predict differences between consecutive time steps
    log_grad_norm : bool, default=False
        Whether to log gradient norms during training for debugging.
    compile_model : bool, default=False
        Whether to compile the model using PyTorch 2.0 with dynamic shapes.
    logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log during training, validation, and testing.
        If empty, no additional metrics will be logged (only the loss from the criterion).
        Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE".

    Attributes
    ----------
    model : VanillaTransformerModel
        The underlying transformer model implementation.
    criterion : QuantileLoss
        The quantile loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models import VanillaTransformer
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create vanilla transformer for point prediction
    >>> model = VanillaTransformer(
    ...     num_features=10,
    ...     ctxt_seq_len=96,     # 4 days of hourly data
    ...     tgt_seq_len=24,      # 1 day prediction horizon
    ...     d_model=256,
    ...     num_heads=8,
    ...     num_encoder_layers=4,
    ...     num_decoder_layers=4,
    ...     dropout=0.1
    ... )
    >>>
    >>> # Create transformer for probabilistic forecasting
    >>> quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    >>> model = VanillaTransformer(
    ...     num_features=10,
    ...     ctxt_seq_len=168,    # 1 week of hourly data
    ...     tgt_seq_len=24,      # 1 day prediction horizon
    ...     criterion=QuantileLoss(quantiles=quantiles),
    ...     compile_model=True
    ... )
    >>>
    >>> # Train with encoder-decoder data
    >>> datamodule = EncoderDecoderDataModule(...)
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)

    Notes
    -----
    **Architecture Overview:**

    The Vanilla Transformer consists of:

    1. **Encoder Stack**: Processes historical context with self-attention
       - Multi-head self-attention layers
       - Position-wise feed-forward networks
       - Residual connections and layer normalization

    2. **Decoder Stack**: Generates predictions autoregressively
       - Masked multi-head self-attention (causal masking)
       - Multi-head cross-attention to encoder outputs
       - Position-wise feed-forward networks
       - Residual connections and layer normalization

    3. **Output Projection**: Maps decoder outputs to prediction space

    **Key Features:**

    - **Self-Attention**: Captures long-range dependencies in time series
    - **Cross-Attention**: Allows decoder to attend to relevant encoder states
    - **Positional Encoding**: Handles temporal ordering information
    - **Causal Masking**: Ensures autoregressive generation properties

    **Input Requirements:**

    The model expects :class:`transformertf.data.EncoderDecoderTargetSample` with:
    - `encoder_input`: Historical context (B, ctxt_seq_len, num_features)
    - `decoder_input`: Future context (B, tgt_seq_len, num_features)

    **Attention Mechanisms:**

    The model uses three types of attention:
    1. Encoder self-attention: Context understanding
    2. Decoder self-attention: Autoregressive generation
    3. Decoder-encoder cross-attention: Context integration

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in
           neural information processing systems 30 (2017).

    See Also
    --------
    TransformerModuleBase : Base class for transformer models
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    TemporalFusionTransformer : Advanced transformer with variable selection
    """

    def __init__(
        self,
        num_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
        criterion: QuantileLoss | None = None,
        prediction_type: typing.Literal["delta", "point"] | None = None,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        if criterion is None:
            criterion = QuantileLoss()
            self.hparams["output_dim"] = len(criterion.quantiles)
            output_dim = self.hparams["output_dim"]
        self.criterion = criterion

        self.model = VanillaTransformerModel(
            num_features=num_features,
            seq_len=ctxt_seq_len,
            out_seq_len=tgt_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            fc_dim=fc_dim,
            output_dim=output_dim,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> dict[str, torch.Tensor]:
        """
        Forward pass through the Vanilla Transformer model.

        This method processes the input batch through the transformer encoder-decoder
        architecture, using the encoder to process historical context and the decoder
        to generate autoregressive predictions.

        Parameters
        ----------
        x : EncoderDecoderTargetSample
            Input batch containing the following keys:
            - "encoder_input": Historical context (B, ctxt_seq_len, num_features)
            - "decoder_input": Future context (B, tgt_seq_len, num_features)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing model outputs:
            - "output": Predictions (B, tgt_seq_len, output_dim)

        Notes
        -----
        **Encoder-Decoder Processing:**

        The forward pass consists of:
        1. **Encoder**: Processes historical context with self-attention
        2. **Decoder**: Generates predictions using masked self-attention
           and cross-attention to encoder outputs

        **Input Mapping:**

        - `encoder_input` → `source`: Historical time series data
        - `decoder_input` → `target`: Future context (may include known covariates)

        **Attention Patterns:**

        - Encoder uses bidirectional self-attention over the full context
        - Decoder uses causal (masked) self-attention to maintain autoregressive properties
        - Cross-attention allows decoder to selectively attend to relevant encoder states

        **Output Shape:**

        - For point prediction: (batch_size, tgt_seq_len, 1)
        - For quantile prediction: (batch_size, tgt_seq_len, num_quantiles)

        Examples
        --------
        >>> batch = {
        ...     "encoder_input": torch.randn(32, 96, 10),   # 4 days of hourly data
        ...     "decoder_input": torch.randn(32, 24, 10)    # 1 day context
        ... }
        >>> model = VanillaTransformer(
        ...     num_features=10,
        ...     ctxt_seq_len=96,
        ...     tgt_seq_len=24
        ... )
        >>> output = model(batch)
        >>> print(output["output"].shape)  # torch.Size([32, 24, 7])
        """
        return self.model(
            source=x["encoder_input"],
            target=x["decoder_input"],
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        model_output = self(batch)

        loss = self.calc_loss(model_output, batch)

        loss_dict = {"loss": loss}
        point_prediction_dict: dict[str, torch.Tensor] = {}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}

        self.common_log_step(loss_dict, "train")

        return loss_dict | {"output": model_output} | point_prediction_dict

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        assert "target" in batch
        model_output = self(batch)

        loss = self.calc_loss(model_output, batch)

        loss_dict = {"loss": loss}
        point_prediction_dict: dict[str, torch.Tensor] = {}
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output)
            point_prediction_dict = {"point_prediction": point_prediction}

        self.common_log_step(loss_dict, "validation")

        return loss_dict | {"output": model_output} | point_prediction_dict
