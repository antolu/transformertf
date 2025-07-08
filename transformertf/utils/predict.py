from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

from ..data import DataModuleBase, EncoderDecoderDataModule, TimeSeriesDataModule
from ..data.dataset import EncoderDecoderPredictDataset
from ..models import LightningModuleBase
from ..models.bwlstm import BWLSTM3
from ..nn import QuantileLoss
from ..utils import ops

__all__ = [
    "predict",
    "predict_encoder_decoder",
    "predict_phylstm",
    "predict_timeseries",
]

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(
    module: LightningModuleBase,
    datamodule: DataModuleBase,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    Perform time series forecasting using trained transformertf models.

    This is the main prediction function that automatically dispatches to the appropriate
    prediction method based on the datamodule and model types. It supports various model
    architectures including encoder-decoder transformers, time series models, and physics-
    informed models like PhyLSTM.

    Parameters
    ----------
    module : LightningModuleBase
        The trained Lightning module containing the model for prediction.
        Should be compatible with the provided datamodule type.
    datamodule : DataModuleBase
        The data module that preprocesses and structures the input data.
        The specific type determines which prediction method is used:

        - TimeSeriesDataModule: Uses predict_timeseries or predict_phylstm
        - EncoderDecoderDataModule: Uses predict_encoder_decoder
    past_covariates : pd.DataFrame
        Historical covariate data used as context for prediction.
        Must contain all columns specified in the datamodule configuration.
        Index should be a time index if temporal ordering is important.
    future_covariates : pd.DataFrame
        Future covariate data for the prediction horizon.
        Must have the same column structure as past_covariates.
        The length determines the prediction horizon.
    past_target : pd.DataFrame | np.ndarray | pd.Series | None, optional
        Historical target values, required for encoder-decoder models.
        Should align temporally with past_covariates. If None, only
        valid for models that don't require target history.
    device : torch.device, optional
        The device to perform predictions on. Defaults to CUDA if available,
        otherwise CPU. The model will be moved to this device automatically.

    Returns
    -------
    np.ndarray
        Predicted target values for the future time horizon.
        Shape is (prediction_length,) for single-target prediction or
        (prediction_length, n_targets) for multi-target prediction.
        Values are in the original scale if inverse transforms are applied.

    Raises
    ------
    NotImplementedError
        If the datamodule type is not supported for prediction.
    AssertionError
        If past_target is None when using EncoderDecoderDataModule.

    Notes
    -----
    This function handles several important aspects automatically:

    - **Model dispatch**: Selects the appropriate prediction method based on
      datamodule and model types
    - **Device handling**: Moves tensors to the specified device
    - **Transform inversion**: Applies inverse transforms if configured
    - **Autoregressive prediction**: For encoder-decoder models, generates
      predictions step-by-step using previous outputs

    The prediction process varies by model type:

    - **TimeSeriesDataModule + BWLSTM3**: Uses physics-informed prediction
      with specialized handling for Bouc-Wen hysteresis models
    - **TimeSeriesDataModule + other models**: Standard time series prediction
      with sliding window approach
    - **EncoderDecoderDataModule**: Autoregressive prediction with encoder-decoder
      architecture, suitable for transformers and sequence-to-sequence models

    Examples
    --------
    Basic prediction with transformer model:

    >>> import pandas as pd
    >>> from transformertf.utils import predict
    >>> from transformertf.models import TransformerLightning
    >>> from transformertf.data import EncoderDecoderDataModule
    >>>
    >>> # Load trained model and datamodule
    >>> model = TransformerLightning.load_from_checkpoint("model.ckpt")
    >>> datamodule = EncoderDecoderDataModule(...)
    >>>
    >>> # Prepare data
    >>> past_covariates = pd.DataFrame({
    ...     "feature_1": [1, 2, 3, 4, 5],
    ...     "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5]
    ... })
    >>> future_covariates = pd.DataFrame({
    ...     "feature_1": [6, 7, 8],
    ...     "feature_2": [0.6, 0.7, 0.8]
    ... })
    >>> past_target = pd.Series([10, 15, 20, 25, 30])
    >>>
    >>> # Generate predictions
    >>> predictions = predict(
    ...     model, datamodule, past_covariates,
    ...     future_covariates, past_target
    ... )
    >>> print(f"Predicted values: {predictions}")

    Physics-informed prediction with PhyLSTM:

    >>> from transformertf.models.bwlstm import BWLSTM3
    >>> from transformertf.data import TimeSeriesDataModule
    >>>
    >>> # PhyLSTM model for structural dynamics
    >>> phylstm_model = BWLSTM3.load_from_checkpoint("phylstm.ckpt")
    >>> ts_datamodule = TimeSeriesDataModule(...)
    >>>
    >>> # Generate physics-informed predictions
    >>> predictions = predict(
    ...     phylstm_model, ts_datamodule,
    ...     past_covariates, future_covariates
    ... )

    GPU prediction with custom device:

    >>> import torch
    >>> device = torch.device("cuda:1")  # Use specific GPU
    >>> predictions = predict(
    ...     model, datamodule, past_covariates,
    ...     future_covariates, past_target, device=device
    ... )

    See Also
    --------
    predict_encoder_decoder : Direct encoder-decoder prediction
    predict_timeseries : Direct time series prediction
    predict_phylstm : Physics-informed LSTM prediction
    transformertf.data.DataModuleBase : Base class for data modules
    transformertf.models.LightningModuleBase : Base class for Lightning modules
    """
    if isinstance(datamodule, TimeSeriesDataModule) and isinstance(module, BWLSTM3):
        return predict_phylstm(
            module,
            datamodule,
            past_covariates,
            future_covariates,
            device,
        )
    if isinstance(datamodule, TimeSeriesDataModule):
        return predict_timeseries(
            module,
            datamodule,
            past_covariates,
            future_covariates,
            device,
        )
    if isinstance(datamodule, EncoderDecoderDataModule):
        assert past_target is not None, "past_target must be provided"
        return predict_encoder_decoder(
            module,
            datamodule,
            past_covariates,
            future_covariates,
            past_target,  # type: ignore
            device=device,
        )
    msg = f"Predicting with datamodule of type {type(datamodule)} is not implemented"
    raise NotImplementedError(msg)


def predict_timeseries(
    module: LightningModuleBase,
    datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    Generate predictions using time series models with sliding window approach.

    This function performs prediction for time series models that process data
    in sliding windows. It concatenates past and future covariates, creates
    overlapping windows, and generates predictions for each window position.

    Parameters
    ----------
    module : LightningModuleBase
        The trained Lightning module for time series prediction.
        Compatible with models like LSTM, GRU, TSMixer, etc.
    datamodule : TimeSeriesDataModule
        Time series data module that handles window generation and preprocessing.
        Must be configured with appropriate window size and stride parameters.
    past_covariates : pd.DataFrame
        Historical covariate data providing context for prediction.
        Should contain all input features specified in datamodule configuration.
    future_covariates : pd.DataFrame
        Future covariate data for the prediction period.
        Must have consistent column structure with past_covariates.
    device : torch.device, optional
        Device for computation. Defaults to CUDA if available, otherwise CPU.

    Returns
    -------
    np.ndarray
        Predicted values for the future horizon.
        Shape: (prediction_length,) for single target prediction.
        Values are inverse-transformed to original scale if transforms are configured.

    Notes
    -----
    **Warning**: This function is marked as UNTESTED and should be used with caution
    in production environments. Thorough testing is recommended before deployment.

    The prediction process involves:

    1. **Data concatenation**: Combines past and future covariates
    2. **Window generation**: Creates sliding windows using datamodule.make_dataset
    3. **Batch prediction**: Processes each window through the model
    4. **Output aggregation**: Concatenates predictions from all windows
    5. **Transform inversion**: Applies inverse transforms if configured
    6. **Truncation**: Limits output to the future prediction horizon

    The function handles downsampling by adjusting the truncation logic
    to account for the downsample factor specified in the datamodule.

    Examples
    --------
    Basic time series prediction:

    >>> import pandas as pd
    >>> from transformertf.utils import predict_timeseries
    >>> from transformertf.models import LSTMLightning
    >>> from transformertf.data import TimeSeriesDataModule
    >>>
    >>> # Setup model and datamodule
    >>> model = LSTMLightning.load_from_checkpoint("lstm_model.ckpt")
    >>> datamodule = TimeSeriesDataModule(
    ...     window_size=100,
    ...     prediction_length=50,
    ...     input_columns=["sensor_1", "sensor_2"],
    ...     target_column="target"
    ... )
    >>>
    >>> # Prepare time series data
    >>> past_data = pd.DataFrame({
    ...     "sensor_1": np.random.randn(200),
    ...     "sensor_2": np.random.randn(200)
    ... })
    >>> future_data = pd.DataFrame({
    ...     "sensor_1": np.random.randn(50),
    ...     "sensor_2": np.random.randn(50)
    ... })
    >>>
    >>> # Generate predictions
    >>> predictions = predict_timeseries(
    ...     model, datamodule, past_data, future_data
    ... )
    >>> print(f"Forecast shape: {predictions.shape}")

    See Also
    --------
    predict : Main prediction dispatcher function
    predict_phylstm : Physics-informed prediction for structural models
    transformertf.data.TimeSeriesDataModule : Time series data module
    transformertf.models.LightningModuleBase : Base Lightning module class
    """
    covariates = pd.concat((past_covariates, future_covariates))

    dataset = datamodule.make_dataset(covariates, predict=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    outputs = []
    for _idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module(batch)
        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        outputs.append(model_output)

    outputs_t = torch.cat([o.squeeze(0) for o in outputs], dim=0)

    if datamodule.target_transform is not None:
        outputs_t = datamodule.target_transform.inverse_transform(
            covariates[datamodule.hparams["input_columns"][0]].to_numpy(),
            outputs_t,
        )

    # truncate the outputs to the length of the future covariates
    outputs_t = outputs_t[: dataset.num_points]
    outputs_t = outputs_t[len(past_covariates) // datamodule.hparams["downsample"] :]
    return typing.cast(torch.Tensor, outputs_t).squeeze().numpy()


def predict_phylstm(
    module: BWLSTM3,
    datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    Generate physics-informed predictions using Bouc-Wen LSTM models.

    This specialized prediction function is designed for the BWLSTM3 model,
    which incorporates physics-based knowledge for structural dynamics and
    hysteresis modeling. It follows the Lightning prediction workflow with
    proper hook calls and handles the model's specific output structure.

    Parameters
    ----------
    module : BWLSTM3
        The trained Bouc-Wen LSTM model (Physics-informed LSTM).
        This model combines LSTM networks with Bouc-Wen hysteresis equations
        for modeling nonlinear structural behavior.
    datamodule : TimeSeriesDataModule
        Time series data module configured for structural dynamics data.
        Should include appropriate scaling and preprocessing for displacement,
        velocity, acceleration, or force measurements.
    past_covariates : pd.DataFrame
        Historical structural response data (e.g., displacement, acceleration).
        Must contain all input features required by the physics model.
    future_covariates : pd.DataFrame
        Future excitation or loading conditions for prediction.
        Used to drive the physics-based forward simulation.
    device : torch.device, optional
        Computation device. Defaults to CUDA if available, otherwise CPU.

    Returns
    -------
    np.ndarray
        Predicted structural response (typically displacement or force).
        Shape: (prediction_length,) representing the hysteretic response.
        Values are in original engineering units after inverse transformation.

    Notes
    -----
    This function implements the complete Lightning prediction workflow:

    1. **Prediction lifecycle hooks**: Calls on_predict_start(), on_predict_epoch_start()
    2. **Batch processing**: Iterates through prediction dataset
    3. **Model prediction**: Uses predict_step() for physics-informed forward pass
    4. **Hook callbacks**: Calls on_predict_batch_end() after each batch
    5. **Cleanup**: Calls on_predict_epoch_end(), on_predict_end()

    The BWLSTM3 model output structure includes:

    - **"z" tensor**: Hysteretic displacement component (B parameter in Bouc-Wen)
    - **Physics constraints**: Enforced through embedded differential equations
    - **Nonlinear dynamics**: Captured through learned hysteresis parameters

    The function specifically extracts the B component (z[..., 0]) which
    represents the primary hysteretic response variable in structural dynamics.

    Examples
    --------
    Structural response prediction for earthquake simulation:

    >>> import pandas as pd
    >>> from transformertf.utils import predict_phylstm
    >>> from transformertf.models.bwlstm import BWLSTM3
    >>> from transformertf.data import TimeSeriesDataModule
    >>>
    >>> # Load trained physics-informed model
    >>> phylstm_model = BWLSTM3.load_from_checkpoint("phylstm_structure.ckpt")
    >>>
    >>> # Configure datamodule for structural data
    >>> datamodule = TimeSeriesDataModule(
    ...     input_columns=["acceleration", "velocity"],
    ...     target_column="displacement",
    ...     window_size=1000,  # Long history for dynamics
    ...     prediction_length=500
    ... )
    >>>
    >>> # Historical structural response
    >>> past_response = pd.DataFrame({
    ...     "acceleration": earthquake_accel_history,
    ...     "velocity": velocity_history
    ... })
    >>>
    >>> # Future earthquake excitation
    >>> future_excitation = pd.DataFrame({
    ...     "acceleration": predicted_earthquake_accel,
    ...     "velocity": np.zeros(500)  # Initial velocity assumption
    ... })
    >>>
    >>> # Predict structural response
    >>> hysteretic_response = predict_phylstm(
    ...     phylstm_model, datamodule,
    ...     past_response, future_excitation
    ... )
    >>>
    >>> print(f"Maximum predicted displacement: {np.max(hysteretic_response):.3f} m")

    Multi-story building response prediction:

    >>> # Configure for multi-DOF system
    >>> datamodule_multistory = TimeSeriesDataModule(
    ...     input_columns=["base_accel", "wind_force"],
    ...     target_column="top_displacement",
    ...     transforms={"top_displacement": scaler}
    ... )
    >>>
    >>> # Predict with environmental loading
    >>> building_response = predict_phylstm(
    ...     phylstm_model, datamodule_multistory,
    ...     historical_data, future_loading
    ... )

    See Also
    --------
    predict : Main prediction dispatcher that automatically selects this function
    predict_timeseries : Standard time series prediction without physics
    transformertf.models.bwlstm.BWLSTM3 : Bouc-Wen LSTM model implementation
    transformertf.data.TimeSeriesDataModule : Time series data handling
    """
    covariates = pd.concat((past_covariates, future_covariates))
    # covariates = covariates.rename(
    #     columns={
    #         cov.name: cov.col
    #         for cov in [*datamodule.known_covariates, datamodule.target_covariate]
    #     }
    # )
    #
    dataset = datamodule.make_dataset(covariates, predict=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    outputs = []
    inputs = []

    module.on_predict_start()
    module.on_predict_epoch_start()
    for idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module.predict_step(batch, idx)
        module.on_predict_batch_end(model_output, batch, idx)  # type: ignore[arg-type]

        model_output = ops.to_cpu(model_output)  # type: ignore[type-var]
        model_output = ops.detach(model_output)  # type: ignore[type-var]

        inputs.append(batch["input"])
        outputs.append(model_output)

    module.on_predict_epoch_end()
    module.on_predict_end()

    predictions = torch.cat([o["output"]["z"].squeeze(0) for o in outputs], dim=0)
    predictions = predictions[..., 0]  # get B

    inputs_t = torch.cat([i.squeeze(0) for i in inputs], dim=0).squeeze()

    if datamodule.target_covariate.name in datamodule.transforms:
        input_transform = datamodule.transforms[datamodule.known_covariates[0].name]

        inputs_t = input_transform.inverse_transform(inputs_t)

        predictions = datamodule.transforms[
            datamodule.target_covariate.name
        ].inverse_transform(
            inputs_t,
            predictions,
        )

    # truncate the outputs to the length of the future covariates
    predictions = predictions[: dataset.num_points]
    predictions = predictions[
        len(past_covariates) // datamodule.hparams["downsample"] :
    ]
    return typing.cast(torch.Tensor, predictions).squeeze().numpy()


@typing.overload
def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    device: torch.device = DEFAULT_DEVICE,
    *,
    raw_output: typing.Literal[False] = False,
) -> np.ndarray: ...


@typing.overload
def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    device: torch.device = DEFAULT_DEVICE,
    *,
    raw_output: typing.Literal[True],
) -> torch.Tensor: ...


def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    device: torch.device = DEFAULT_DEVICE,
    *,
    raw_output: bool = False,
) -> np.ndarray | torch.Tensor:
    """
    Generate autoregressive predictions using encoder-decoder model architectures.

    This function implements autoregressive forecasting for encoder-decoder models
    like Transformers, Temporal Fusion Transformers, and other sequence-to-sequence
    architectures. It encodes the historical context and generates future predictions
    step-by-step, using previous predictions as inputs for subsequent steps.

    Parameters
    ----------
    module : LightningModuleBase
        The trained encoder-decoder model (e.g., Transformer, TFT).
        Must be compatible with EncoderDecoderDataModule and support
        autoregressive generation through proper input/output handling.
    datamodule : EncoderDecoderDataModule
        Data module configured for encoder-decoder prediction.
        Handles encoding sequences, target alignment, and preprocessing.
    past_covariates : pd.DataFrame
        Historical covariate data used by the encoder for context.
        Must contain all features specified in known_covariates configuration.
    future_covariates : pd.DataFrame
        Future covariate data used by the decoder during generation.
        Length determines the prediction horizon. Must include known future
        information like calendar features, planned interventions, etc.
    past_target : pd.DataFrame | np.ndarray | pd.Series
        Historical target values that provide context for autoregressive prediction.
        Must align temporally with past_covariates and contain the target variable.
    device : torch.device, optional
        Device for computation. Defaults to CUDA if available, otherwise CPU.
        Model and data tensors are automatically moved to this device.
    raw_output : bool, optional
        If False (default), returns processed point predictions in original scale.
        If True, returns raw model outputs without post-processing or inverse transforms.
        Useful for accessing prediction intervals or distribution parameters.

    Returns
    -------
    np.ndarray | torch.Tensor
        Predicted values for the future horizon.

        - When raw_output=False: Returns np.ndarray with shape (prediction_length,)
          containing point predictions in original scale
        - When raw_output=True: Returns torch.Tensor with raw model outputs,
          shape depends on model architecture and may include distribution parameters

    Notes
    -----
    The autoregressive prediction process involves:

    1. **Data preprocessing**: Applies configured transforms and scaling
    2. **Encoder context**: Processes historical data to create context representations
    3. **Autoregressive decoding**: Generates predictions step-by-step:

       - Uses encoder context and previous targets
       - Predicts next time step
       - Appends prediction to target history
       - Repeats until full horizon is generated

    4. **Post-processing**: Converts outputs to point predictions and applies inverse transforms

    The function supports various model types:

    - **Transformers**: Attention-based sequence modeling
    - **Temporal Fusion Transformer**: Interpretable forecasting with attention
    - **Custom encoder-decoder**: Any model following the encoder-decoder pattern

    For quantile regression models, the function automatically extracts point
    predictions from the quantile outputs using the median or mean quantile.

    Examples
    --------
    Standard transformer forecasting:

    >>> import pandas as pd
    >>> from transformertf.utils import predict_encoder_decoder
    >>> from transformertf.models import TransformerLightning
    >>> from transformertf.data import EncoderDecoderDataModule
    >>>
    >>> # Load trained transformer model
    >>> transformer = TransformerLightning.load_from_checkpoint("transformer.ckpt")
    >>>
    >>> # Configure encoder-decoder datamodule
    >>> datamodule = EncoderDecoderDataModule(
    ...     ctxt_seq_len=168,  # 1 week context
    ...     tgt_seq_len=24,    # 1 day prediction
    ...     known_covariates=["hour", "day_of_week", "temperature"],
    ...     target_covariate="energy_demand"
    ... )
    >>>
    >>> # Prepare historical data
    >>> past_data = pd.DataFrame({
    ...     "hour": range(168),
    ...     "day_of_week": [i % 7 for i in range(168)],
    ...     "temperature": np.random.normal(20, 5, 168)
    ... })
    >>> past_targets = pd.Series(np.random.normal(1000, 200, 168))
    >>>
    >>> # Future covariates (known information)
    >>> future_data = pd.DataFrame({
    ...     "hour": range(168, 192),
    ...     "day_of_week": [i % 7 for i in range(168, 192)],
    ...     "temperature": np.random.normal(22, 3, 24)  # Weather forecast
    ... })
    >>>
    >>> # Generate predictions
    >>> predictions = predict_encoder_decoder(
    ...     transformer, datamodule,
    ...     past_data, future_data, past_targets
    ... )
    >>> print(f"24-hour forecast: {predictions}")

    Accessing raw model outputs for uncertainty quantification:

    >>> # Get raw outputs with prediction intervals
    >>> raw_outputs = predict_encoder_decoder(
    ...     transformer, datamodule,
    ...     past_data, future_data, past_targets,
    ...     raw_output=True
    ... )
    >>>
    >>> # Extract quantiles for uncertainty bounds
    >>> if raw_outputs.shape[-1] > 1:  # Multi-quantile output
    ...     lower_bound = raw_outputs[..., 0]  # 10th percentile
    ...     median = raw_outputs[..., 1]       # 50th percentile
    ...     upper_bound = raw_outputs[..., 2]  # 90th percentile

    Multi-variate forecasting:

    >>> # Configure for multiple targets
    >>> datamodule_multi = EncoderDecoderDataModule(
    ...     target_covariate=["demand", "price"],
    ...     ctxt_seq_len=168,
    ...     tgt_seq_len=24
    ... )
    >>>
    >>> multi_predictions = predict_encoder_decoder(
    ...     multi_target_model, datamodule_multi,
    ...     past_covariates, future_covariates, multi_past_targets
    ... )
    >>> print(f"Multi-target forecast shape: {multi_predictions.shape}")

    See Also
    --------
    predict : Main prediction dispatcher function
    to_point_prediction : Convert model outputs to point predictions
    transformertf.data.EncoderDecoderDataModule : Encoder-decoder data handling
    transformertf.data.dataset.EncoderDecoderPredictDataset : Prediction dataset
    transformertf.models.transformer.TransformerLightning : Transformer implementation
    transformertf.models.temporal_fusion_transformer : TFT implementation
    """
    if isinstance(past_target, pd.DataFrame):
        past_target = past_target[datamodule.hparams["target_covariate"]].to_numpy()
    elif isinstance(past_target, pd.Series):
        past_target = past_target.to_numpy()

    known_past_columns = datamodule.hparams.get("known_past_covariates")

    past_covariates["target"] = past_target
    past_df = datamodule.preprocess_dataframe(past_covariates)

    past_covariates = past_df[past_covariates.columns]
    (past_df[known_past_columns] if known_past_columns is not None else None)
    past_target = past_df["target"].to_numpy()
    future_covariates = datamodule.preprocess_dataframe(future_covariates)

    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=datamodule.hparams["ctxt_seq_len"],
        prediction_length=datamodule.hparams["tgt_seq_len"],
        transforms=datamodule.transforms,
        input_columns=datamodule.hparams["known_covariates"],
        target_column=datamodule.hparams["target_covariate"],
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    outputs = []
    for idx, batch in enumerate(dataloader):
        batch["encoder_lengths"] = torch.ones(
            [batch["encoder_input"].shape[0], 1],
            dtype=batch["encoder_input"].dtype,
        )
        batch = ops.to(batch, device)

        model_output = module(batch)
        if isinstance(model_output, dict) and "output" in model_output:
            model_output = model_output["output"]
        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        outputs.append(model_output)

        if idx < len(dataloader) - 1:
            point_prediction = to_point_prediction(
                model_output, module.criterion
            ).squeeze()
            dataset.append_past_target(point_prediction)

    outputs_t = torch.cat([o.squeeze(0) for o in outputs], dim=0)

    if raw_output:
        return outputs_t.numpy()  # type: ignore[attr-defined]

    outputs_t = to_point_prediction(outputs_t, module.criterion)

    # truncate the outputs to the length of the future covariates
    outputs_t = outputs_t[: len(future_covariates)]

    if datamodule.target_covariate.name in datamodule.transforms:
        future_x = future_covariates[datamodule.known_covariates[0].name]
        outputs_t = datamodule.transforms[
            datamodule.target_covariate.name
        ].inverse_transform(future_x, outputs_t)

    # truncate the outputs to the length of the future covariates
    return outputs_t[: len(future_covariates)].numpy()  # type: ignore[attr-defined]


def to_point_prediction(
    model_output: torch.Tensor, criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Convert model outputs to point predictions based on the loss criterion.

    This utility function handles the conversion of various model output formats
    to scalar point predictions. It's particularly useful for models that output
    distributions, quantiles, or other complex prediction formats.

    Parameters
    ----------
    model_output : torch.Tensor
        Raw model outputs to convert to point predictions.
        Shape: (..., n_outputs) where n_outputs depends on the model type:

        - Standard regression: (..., 1) single values
        - Quantile regression: (..., n_quantiles) multiple quantiles
        - Distributional: (..., n_parameters) distribution parameters
    criterion : torch.nn.Module
        The loss criterion used during training, which determines how to
        extract point predictions from the model outputs.

        - QuantileLoss: Extracts median/mean from quantile outputs
        - MSELoss/MAELoss: Returns outputs directly
        - Custom criteria: May have specific point prediction methods

    Returns
    -------
    torch.Tensor
        Point predictions extracted from model outputs.
        Shape: (...,) with the same leading dimensions as input but
        reduced to scalar predictions along the last dimension.

    Raises
    ------
    ValueError
        If model_output has fewer than 2 dimensions, which is required
        for proper tensor operations.

    Notes
    -----
    The conversion strategy depends on the criterion type:

    - **QuantileLoss**: Uses the criterion's built-in point_prediction method
      which typically selects the median quantile or computes a weighted average
    - **Standard losses**: Returns the model output directly, assuming it
      represents point predictions

    For quantile regression, the point prediction is usually the 50th percentile
    (median) when available, or a weighted combination of all quantiles.

    The function handles both 2D and higher-dimensional tensors, preserving
    batch dimensions while reducing the output dimension.

    Examples
    --------
    Converting quantile regression outputs:

    >>> import torch
    >>> from transformertf.utils.predict import to_point_prediction
    >>> from transformertf.nn import QuantileLoss
    >>>
    >>> # Quantile model outputs [10th, 50th, 90th percentiles]
    >>> quantile_outputs = torch.tensor([
    ...     [8.5, 10.0, 11.5],  # First prediction
    ...     [9.0, 11.2, 13.1]   # Second prediction
    ... ])
    >>>
    >>> # Quantile loss criterion
    >>> quantile_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    >>>
    >>> # Extract point predictions (median)
    >>> point_preds = to_point_prediction(quantile_outputs, quantile_loss)
    >>> print(point_preds)  # tensor([10.0, 11.2])

    Standard regression outputs:

    >>> import torch.nn as nn
    >>>
    >>> # Standard regression outputs
    >>> regression_outputs = torch.tensor([[10.5], [11.8]])
    >>> mse_loss = nn.MSELoss()
    >>>
    >>> # Pass through directly
    >>> point_preds = to_point_prediction(regression_outputs, mse_loss)
    >>> print(point_preds)  # tensor([10.5, 11.8])

    Batch processing with 3D tensors:

    >>> # Batch of sequence predictions
    >>> batch_outputs = torch.randn(32, 24, 3)  # [batch, time, quantiles]
    >>> point_preds = to_point_prediction(batch_outputs, quantile_loss)
    >>> print(point_preds.shape)  # torch.Size([32, 24])

    See Also
    --------
    predict_encoder_decoder : Uses this function for post-processing
    transformertf.nn.QuantileLoss : Quantile regression loss implementation
    transformertf.nn.QuantileLoss.point_prediction : Method for extracting point predictions
    """
    if model_output.ndim < 2:
        msg = f"Model output must have at least 2 dimensions, got {model_output.ndim}"
        raise ValueError(msg)
    if isinstance(criterion, QuantileLoss):
        if model_output.ndim == 2:
            return criterion.point_prediction(model_output.unsqueeze(0)).squeeze(0)
        return criterion.point_prediction(model_output)

    return model_output
