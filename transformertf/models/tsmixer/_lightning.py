from __future__ import annotations

import collections.abc
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_module import DEFAULT_LOGGING_METRICS, LightningModuleBase, MetricLiteral
from ._model import TSMixerModel


class TSMixer(LightningModuleBase):
    """
    TSMixer model for multivariate time series forecasting.

    TSMixer (Time Series Mixer) is a simple yet effective MLP-based architecture
    for time series forecasting that uses mixing layers to capture temporal and
    cross-variate interactions. The model alternates between time-mixing and
    feature-mixing operations to learn complex patterns in multivariate time series.

    The architecture is inspired by the MLP-Mixer design but adapted specifically
    for time series data, providing an efficient alternative to transformer-based
    models while maintaining competitive performance.

    Parameters
    ----------
    num_features : int
        Number of input features (time series variables).
    num_static_features : int, default=0
        Number of static features. Currently not used in the model.
    ctxt_seq_len : int, default=500
        Length of the context (encoder) sequence for historical data.
    tgt_seq_len : int, default=300
        Length of the target (prediction) sequence.
    fc_dim : int, default=1024
        Dimension of the fully connected layers in the mixing blocks.
        Higher values increase model capacity.
    n_dim_model : int or None, default=None
        Hidden dimension of the model. If None, uses num_features.
    num_blocks : int, default=4
        Number of TSMixer blocks. Each block contains time-mixing and
        feature-mixing layers.
    dropout : float, default=0.1
        Dropout probability applied throughout the model.
    activation : {"relu", "gelu"}, default="relu"
        Activation function used in the mixing layers.
    norm : {"batch", "layer"}, default="batch"
        Type of normalization to use in the mixing blocks.
    criterion : QuantileLoss or torch.nn.MSELoss or torch.nn.HuberLoss or None, default=None
        Loss function for training. If None, defaults to MSELoss.
        Use QuantileLoss for probabilistic forecasting.
    log_grad_norm : bool, default=False
        Whether to log gradient norms during training.
    compile_model : bool, default=False
        Whether to compile the model using PyTorch 2.0.
    logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log during training, validation, and testing.
        If empty, no additional metrics will be logged (only the loss from the criterion).
        Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE".

    Attributes
    ----------
    model : TSMixerModel
        The underlying TSMixer model implementation.
    criterion : torch.nn.Module
        The loss function used for training.
    hparams : dict
        Hyperparameters stored by Lightning.

    Examples
    --------
    >>> from transformertf.models import TSMixer
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.nn import QuantileLoss
    >>> import lightning as L
    >>>
    >>> # Create TSMixer model for point prediction
    >>> model = TSMixer(
    ...     num_features=10,
    ...     ctxt_seq_len=168,    # 1 week of hourly data
    ...     tgt_seq_len=24,      # 1 day prediction horizon
    ...     fc_dim=512,
    ...     num_blocks=6,
    ...     activation="gelu",
    ...     norm="layer"
    ... )
    >>>
    >>> # Create TSMixer for probabilistic forecasting
    >>> quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    >>> model = TSMixer(
    ...     num_features=10,
    ...     ctxt_seq_len=168,
    ...     tgt_seq_len=24,
    ...     criterion=QuantileLoss(quantiles=quantiles),
    ...     compile_model=True
    ... )
    >>>
    >>> # Train with encoder-decoder data
    >>> datamodule = EncoderDecoderDataModule(...)
    >>> trainer = L.Trainer(max_epochs=50)
    >>> trainer.fit(model, datamodule)

    Notes
    -----
    **Architecture Overview:**

    The TSMixer model consists of:

    1. **Input Projection**: Maps input features to model dimension
    2. **TSMixer Blocks**: Alternating time-mixing and feature-mixing layers
       - Time-mixing: Captures temporal patterns across time steps
       - Feature-mixing: Captures cross-variate interactions between features
    3. **Output Projection**: Maps to final prediction dimension

    **Key Advantages:**

    - **Simplicity**: Pure MLP-based architecture, easier to understand and debug
    - **Efficiency**: Generally faster than transformer models
    - **Effectiveness**: Competitive performance on many time series tasks
    - **Scalability**: Handles multivariate time series well

    **Input Requirements:**

    The model expects :class:`transformertf.data.EncoderDecoderTargetSample` with:
    - `encoder_input`: Past covariates (B, ctxt_seq_len, num_features)
    - `decoder_input`: Future covariates (B, tgt_seq_len, num_features)
      Note: Last feature is treated as target, so only the first num_features-1
      features are used as future covariates.

    **Feature Mixing:**

    The model assumes the last feature in decoder_input is the target variable
    and uses only the first num_features-1 features as future covariates.

    References
    ----------
    .. [1] Chen, Si-An, et al. "TSMixer: An All-MLP Architecture for Time Series
           Forecasting." arXiv preprint arXiv:2303.06053 (2023).

    See Also
    --------
    LightningModuleBase : Base class for all models
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    """

    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        ctxt_seq_len: int = 500,
        tgt_seq_len: int = 300,
        fc_dim: int = 1024,
        n_dim_model: int | None = None,
        num_blocks: int = 4,
        dropout: float = 0.1,
        activation: typing.Literal["relu", "gelu"] = "relu",
        norm: typing.Literal["batch", "layer"] = "batch",
        criterion: (QuantileLoss | torch.nn.MSELoss | torch.nn.HuberLoss | None) = None,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        self.criterion = criterion or torch.nn.MSELoss()

        output_dim = 1
        if isinstance(self.criterion, QuantileLoss):
            output_dim = len(self.criterion.quantiles)

        self.hparams["output_dim"] = output_dim

        self.model = TSMixerModel(
            num_feat=num_features,
            num_future_feat=num_features - 1,
            num_static_real_feat=num_static_features,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            fc_dim=fc_dim,
            hidden_dim=n_dim_model,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
            output_dim=output_dim,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> dict[str, torch.Tensor]:
        """
        Forward pass through the TSMixer model.

        This method processes the input batch through the TSMixer architecture,
        extracting past and future covariates and returning predictions.

        Parameters
        ----------
        x : EncoderDecoderTargetSample
            Input batch containing the following keys:
            - "encoder_input": Past covariates (B, ctxt_seq_len, num_features)
            - "decoder_input": Future covariates (B, tgt_seq_len, num_features)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing model outputs:
            - "output": Predictions (B, tgt_seq_len, output_dim)

        Notes
        -----
        **Future Covariates Handling:**

        The method extracts only the first num_features-1 features from the
        decoder input as future covariates, assuming the last feature is the
        target variable:

        .. code-block:: python

            future_covariates = decoder_input[..., :-1]

        This design allows the model to use known future information (like
        calendar features, planned events) while predicting the target variable.

        **Input Shape Requirements:**

        - encoder_input: (batch_size, ctxt_seq_len, num_features)
        - decoder_input: (batch_size, tgt_seq_len, num_features)

        **Output Shape:**

        - output: (batch_size, tgt_seq_len, output_dim)
          - For point prediction: output_dim = 1
          - For quantile prediction: output_dim = number of quantiles

        Examples
        --------
        >>> batch = {
        ...     "encoder_input": torch.randn(32, 168, 10),    # 10 features
        ...     "decoder_input": torch.randn(32, 24, 10)      # same 10 features
        ... }
        >>> model = TSMixer(num_features=10, ctxt_seq_len=168, tgt_seq_len=24)
        >>> output = model(batch)
        >>> print(output["output"].shape)  # torch.Size([32, 24, 1])
        >>>
        >>> # The model uses 9 future covariates (features 0-8)
        >>> # and predicts feature 9 (the target)
        """
        return self.model(
            past_covariates=x["encoder_input"],
            future_covariates=x["decoder_input"][..., :-1],
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single training step for the TSMixer model.

        This method processes a training batch through the TSMixer architecture,
        computes the loss, and returns the results for backpropagation.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Training batch containing encoder inputs, decoder inputs, and targets.
            Expected keys: "encoder_input", "decoder_input", "target".
        batch_idx : int
            Index of the current batch within the training epoch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": The computed training loss
            - "output": The model predictions

        Notes
        -----
        The training step:
        1. Processes input through the TSMixer model
        2. Computes loss using the specified criterion
        3. Logs training metrics via `common_log_step`
        4. Returns loss and output for Lightning's training loop

        The target tensor is expected to have shape (batch_size, tgt_seq_len, 1)
        for univariate forecasting or (batch_size, tgt_seq_len, output_dim) for
        quantile forecasting.
        """
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))

        self.common_log_step({"loss": loss}, "train")

        return {"loss": loss, "output": model_output}

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single validation step for the TSMixer model.

        This method processes a validation batch, computes the model output and loss,
        and extracts point predictions for metric calculation.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            Validation batch containing encoder inputs, decoder inputs, and targets.
            Expected keys: "encoder_input", "decoder_input", "target".
        batch_idx : int
            Index of the current batch within the validation epoch.
        dataloader_idx : int, default=0
            Index of the dataloader when multiple validation dataloaders are used.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": The computed validation loss
            - "output": The raw model predictions
            - "point_prediction": Point estimate for metrics calculation

        Notes
        -----
        The validation step:
        1. Processes input through the TSMixer model
        2. Computes loss using the specified criterion
        3. Extracts point predictions (median for quantile models)
        4. Logs validation metrics via `common_log_step`
        5. Returns outputs for collection by the base class

        For quantile models, the point prediction is the median quantile,
        extracted using `criterion.point_prediction()` and reshaped to
        maintain consistency with the output format.

        The validation outputs are automatically collected by the base class
        and can be accessed via the `validation_outputs` property.
        """
        assert "target" in batch
        target = batch["target"]

        model_output = self(batch)

        loss = typing.cast(torch.Tensor, self.criterion(model_output, target))
        loss_dict = {"loss": loss}

        point_prediction = model_output
        if isinstance(self.criterion, QuantileLoss):
            point_prediction = self.criterion.point_prediction(model_output).unsqueeze(
                -1
            )

        self.common_log_step(loss_dict, "validation")

        return {
            "loss": loss,
            "output": model_output,
            "point_prediction": point_prediction,
        }
