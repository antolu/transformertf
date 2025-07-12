from __future__ import annotations

import collections
import collections.abc
import logging
import typing

import lightning as L
import lightning.pytorch.utilities
import torch

from ..data import EncoderDecoderTargetSample, TimeSeriesSample
from ..nn import QuantileLoss
from ..nn import functional as F
from ..utils import ops

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="LightningModuleBase")
    from .typing import MODEL_OUTPUT, STEP_OUTPUT


log = logging.getLogger(__name__)


# Define available metrics as literals for type safety
MetricLiteral = typing.Literal["MSE", "MAE", "MAPE", "SMAPE", "RMSE"]
DEFAULT_LOGGING_METRICS: tuple[MetricLiteral, ...] = (
    "MSE",
    "MAE",
    "MAPE",
    "SMAPE",
    "RMSE",
)


def setup_criterion_and_output_dim(
    criterion: torch.nn.Module | None,
    output_dim: int,
    default_quantiles: list[float] | None = None,
) -> tuple[torch.nn.Module, int]:
    """
    Set up criterion and adjust output_dim for QuantileLoss compatibility.

    This function provides shared logic for criterion setup across different models,
    ensuring consistent behavior when using QuantileLoss with automatic output_dim
    adjustment.

    Parameters
    ----------
    criterion : torch.nn.Module or None
        The loss function to use. If None, defaults to QuantileLoss for output_dim > 1
        or MSELoss for output_dim = 1.
    output_dim : int
        Desired output dimension. Will be adjusted to match QuantileLoss quantiles.
    default_quantiles : list[float] or None, default=None
        Default quantiles to use when creating QuantileLoss. If None, uses [0.1, 0.5, 0.9].

    Returns
    -------
    tuple[torch.nn.Module, int]
        Tuple of (criterion, adjusted_output_dim)

    Examples
    --------
    >>> criterion, output_dim = setup_criterion_and_output_dim(None, 7)
    >>> isinstance(criterion, QuantileLoss)
    True
    >>> output_dim
    3

    >>> criterion, output_dim = setup_criterion_and_output_dim(None, 1)
    >>> isinstance(criterion, torch.nn.MSELoss)
    True
    >>> output_dim
    1
    """
    if criterion is None:
        if output_dim > 1:
            if default_quantiles is None:
                criterion = QuantileLoss()  # Use QuantileLoss default quantiles
            else:
                criterion = QuantileLoss(quantiles=default_quantiles)
        else:
            criterion = torch.nn.MSELoss()

    if isinstance(criterion, QuantileLoss):
        output_dim = len(criterion.quantiles)

    return criterion, output_dim


class LightningModuleBase(L.LightningModule):
    """
    Base class for all time series forecasting models in the transformertf library.

    This class provides common functionality for PyTorch Lightning modules used in time series
    forecasting tasks. It handles output collection, metric calculation, model compilation,
    and provides a standardized interface for training, validation, and testing procedures.

    The class is designed to work with :class:`transformertf.data.DataModuleBase` and its
    subclasses, providing seamless integration with the data processing pipeline.

    Parameters
    ----------
    logging_metrics : collections.abc.Container[MetricLiteral], default=DEFAULT_LOGGING_METRICS
        Container of metric names to compute and log during training, validation, and testing.
        If empty, no additional metrics will be logged (only the loss from the criterion).
        Available metrics: "MSE", "MAE", "MAPE", "SMAPE", "RMSE".

    Attributes
    ----------
    model : torch.nn.Module
        The main neural network model. Must be set by subclasses.
    criterion : torch.nn.Module
        The loss function used for training. Must be set by subclasses.
    hparams : dict
        Hyperparameters dictionary automatically managed by Lightning.
    validation_outputs : dict[int, list[MODEL_OUTPUT]]
        Collected outputs from validation steps, indexed by dataloader index.
    test_outputs : dict[int, list[MODEL_OUTPUT]]
        Collected outputs from test steps, indexed by dataloader index.
    inference_outputs : dict[int, list[MODEL_OUTPUT]]
        Collected outputs from inference/prediction steps, indexed by dataloader index.

    Examples
    --------
    >>> from transformertf.models import LightningModuleBase
    >>> from transformertf.data import TimeSeriesDataModule
    >>> import torch
    >>>
    >>> class CustomModel(LightningModuleBase):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.model = torch.nn.Linear(10, 1)
    ...         self.criterion = torch.nn.MSELoss()
    ...
    ...     def forward(self, x):
    ...         return self.model(x)
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         # Implementation here
    ...         pass
    >>>
    >>> # Usage with default metrics
    >>> model = CustomModel()
    >>> datamodule = TimeSeriesDataModule(...)
    >>> trainer = L.Trainer()
    >>> trainer.fit(model, datamodule)
    >>>
    >>> # Custom metrics selection
    >>> model = CustomModel(logging_metrics=["MSE", "MAE"])
    >>>
    >>> # No additional metrics (only loss)
    >>> model = CustomModel(logging_metrics=[])

    Notes
    -----
    This base class provides several key features:

    1. **Output Collection**: Automatically collects outputs from validation, test, and
       inference steps for later analysis or visualization.

    2. **Metric Calculation**: Automatically calculates configurable time series metrics
       based on the `logging_metrics` parameter. Default metrics include MSE, MAE, MAPE,
       SMAPE, and RMSE.

    3. **Model Compilation**: Supports PyTorch 2.0 model compilation via the
       `compile_model` hyperparameter for improved performance.

    4. **Gradient Logging**: Optional gradient norm logging via the `log_grad_norm`
       hyperparameter for debugging and monitoring.

    The class expects the model outputs to be either tensors or dictionaries containing
    at least an "output" key. For probabilistic models, a "point_prediction" key can
    be provided to specify the point estimate used for metric calculation.

    See Also
    --------
    TransformerModuleBase : Base class for transformer-based models
    transformertf.data.DataModuleBase : Base class for data modules
    transformertf.nn.functional : Functional utilities for loss calculation
    """

    model: torch.nn.Module
    criterion: torch.nn.Module

    def __init__(
        self,
        *,
        logging_metrics: collections.abc.Container[
            MetricLiteral
        ] = DEFAULT_LOGGING_METRICS,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Store metrics configuration - use from hparams if available, fallback to parameter
        self._logging_metrics = getattr(
            self.hparams, "logging_metrics", logging_metrics
        )

        self._train_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._val_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._test_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._inference_outputs: dict[int, list[MODEL_OUTPUT]] = {}

    def on_fit_start(self) -> None:
        self._validate_hyperparameters()
        self.maybe_compile_model()

    def maybe_compile_model(self) -> None:
        """
        Compile the model using PyTorch 2.0 compilation if enabled in hyperparameters.

        This method checks if the "compile_model" hyperparameter is set to True and,
        if so, applies `torch.compile()` to all child modules except those with "loss"
        in their name. Model compilation can significantly improve performance for
        compatible models and hardware.

        The compilation is applied to all child modules (stored as attributes) that
        don't contain "loss" in their name, allowing loss functions to remain
        uncompiled for compatibility.

        Notes
        -----
        - Requires PyTorch 2.0+ for compilation support
        - Only compiles modules that don't have "loss" in their attribute name
        - Compilation state is handled automatically in `state_dict()` method
        - Best performance gains are typically seen with newer hardware (A100, H100)

        Examples
        --------
        >>> model = SomeModel(compile_model=True)
        >>> # Compilation happens automatically during fit start
        >>> trainer.fit(model, datamodule)
        """
        if self.hparams.get("compile_model"):
            for name, mod in self.named_children():
                if "loss" in name.lower():
                    continue
                setattr(self, name, torch.compile(mod))

    def on_train_start(self) -> None:
        self._train_outputs = {}

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = {}

        super().on_validation_epoch_start()

    def on_train_batch_end(
        self,
        outputs: torch.Tensor | typing.Mapping[str, typing.Any] | None,
        batch: TimeSeriesSample | EncoderDecoderTargetSample,
        batch_idx: int,
    ) -> None:
        if outputs is None:
            msg = "Expected outputs to be provided in training batch end, got None"
            raise ValueError(msg)

        # Validate batch structure
        self._validate_batch_structure(batch, "training")

        if "target" not in batch:
            msg = "Expected 'target' key in batch during training"
            raise ValueError(msg)

        if not isinstance(outputs, dict):
            msg = f"Expected outputs to be dict during training, got {type(outputs)}"
            raise TypeError(msg)

        if self._logging_metrics:
            other_metrics = self.calc_other_metrics(outputs, batch["target"])
            self.common_log_step(other_metrics, "train")  # type: ignore[attr-defined]

        return super().on_train_batch_end(outputs, batch, batch_idx)  # type: ignore[misc]

    def on_validation_batch_end(  # type: ignore[override]
        self,
        outputs: STEP_OUTPUT,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Validate batch structure
        self._validate_batch_structure(batch, "validation")

        if dataloader_idx not in self._val_outputs:
            self._val_outputs[dataloader_idx] = []

        if isinstance(outputs, dict):
            data = outputs | batch  # type: ignore[operator]
        else:
            data = {"output": outputs} | batch

        self._val_outputs[dataloader_idx].append(ops.to_cpu(ops.detach(data)))  # type: ignore[arg-type,type-var]

        if self._logging_metrics:
            if "target" not in batch:
                msg = "Expected 'target' key in batch during validation"
                raise ValueError(msg)
            other_metrics = self.calc_other_metrics(outputs, batch["target"])
            self.common_log_step(other_metrics, "validation")  # type: ignore[attr-defined]

    def on_test_epoch_start(self) -> None:
        self._test_outputs = {}

        super().on_test_epoch_start()

    def on_test_batch_end(  # type: ignore[override]
        self,
        outputs: STEP_OUTPUT,
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._test_outputs:
            self._test_outputs[dataloader_idx] = []

        if isinstance(outputs, dict):
            data = outputs | batch  # type: ignore[operator]
        else:
            data = {"output": outputs} | batch

        self._test_outputs[dataloader_idx].append(ops.to_cpu(ops.detach(data)))  # type: ignore[type-var, arg-type]

    def on_predict_epoch_start(self) -> None:
        self._inference_outputs = {}

        super().on_predict_epoch_start()

    def on_predict_batch_end(  # type: ignore[override]
        self,
        outputs: STEP_OUTPUT,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._inference_outputs:
            self._inference_outputs[dataloader_idx] = []

        if isinstance(outputs, dict):
            data = outputs | batch
        else:
            data = {"output": outputs} | batch

        self._inference_outputs[dataloader_idx].append(ops.to_cpu(ops.detach(data)))  # type: ignore[type-var, arg-type]

    def common_log_step(
        self,
        loss: dict[str, torch.Tensor],
        stage: typing.Literal["train", "validation", "test", "predict"],
    ) -> dict[str, torch.Tensor]:
        """
        Log metrics for a given training stage with consistent formatting.

        This method standardizes the logging of metrics across different training stages
        (train, validation, test, predict) by prefixing metric names with the stage name
        and configuring appropriate logging behavior for each stage.

        Parameters
        ----------
        loss : dict[str, torch.Tensor]
            Dictionary of metric names and their corresponding tensor values.
            Common keys include "loss", "MSE", "MAE", "MAPE", "SMAPE", "RMSE".
        stage : {"train", "validation", "test", "predict"}
            The current training stage, used to prefix metric names and configure
            logging behavior.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of formatted metric names (prefixed with stage) and their
            detached tensor values.

        Examples
        --------
        >>> metrics = {"loss": loss_tensor, "MSE": mse_tensor}
        >>> logged_metrics = self.common_log_step(metrics, "train")
        >>> # Results in {"train/loss": loss_tensor, "train/MSE": mse_tensor}

        Notes
        -----
        The logging behavior varies by stage:
        - **train**: Logged on every step and shown in progress bar
        - **validation/test/predict**: Logged only at epoch end
        - All stages use distributed synchronization (`sync_dist=True`)
        """
        log_dict = {f"{stage}/{k}": v.detach() for k, v in loss.items()}

        if self.logger is not None:
            self.log_dict(
                log_dict,
                on_step=stage == "train",
                prog_bar=stage == "train",
                sync_dist=True,
            )

        return log_dict

    def calc_other_metrics(
        self,
        outputs: STEP_OUTPUT,
        target: torch.Tensor,  # type: ignore[override]
    ) -> dict[str, torch.Tensor]:
        """
        Calculate configurable metrics from the outputs of the model.

        Parameters
        ----------
        outputs : STEP_OUTPUT
            The outputs of the model. Should contain keys "output" and "loss", and optionally "point_prediction".
            If the "point_prediction" key is present, it should be a tensor of shape (batch_size, prediction_horizon, 1),
            otherwise it will be calculated from the "output" tensor.
        target : torch.Tensor
            The target tensor. Should be a tensor of shape (batch_size, prediction_horizon, 1).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the calculated metrics based on metrics_config.

        Raises
        ------
        TypeError
            If outputs or target have invalid types.
        ValueError
            If tensor shapes are incompatible.
        """
        # Validate inputs with helpful error messages
        self._validate_tensor_shapes(outputs, target, "metric calculation")

        loss_dict: dict[str, torch.Tensor] = {}

        # Extract prediction tensor (validation ensures this is safe)
        if "point_prediction" in outputs:
            prediction = outputs["point_prediction"]
        else:
            prediction = outputs["output"]

        target = target.squeeze()
        prediction = prediction.squeeze()

        # Calculate only the metrics specified in configuration
        for metric_name in self._logging_metrics:
            if metric_name == "MSE":
                loss_dict["MSE"] = torch.nn.functional.mse_loss(prediction, target)
            elif metric_name == "MAE":
                loss_dict["MAE"] = torch.nn.functional.l1_loss(prediction, target)
            elif metric_name == "MAPE":
                loss_dict["MAPE"] = F.mape_loss(prediction, target)
            elif metric_name == "SMAPE":
                loss_dict["SMAPE"] = F.smape_loss(prediction, target)
            elif metric_name == "RMSE":
                # Calculate MSE first if needed for RMSE
                if "MSE" not in loss_dict:
                    mse_value = torch.nn.functional.mse_loss(prediction, target)
                    loss_dict["RMSE"] = torch.sqrt(mse_value)
                else:
                    loss_dict["RMSE"] = torch.sqrt(loss_dict["MSE"])

        return loss_dict

    def _validate_hyperparameters(self) -> None:
        """
        Validate hyperparameter configuration for common issues.

        This method checks for common configuration mistakes and provides
        helpful error messages with suggestions for fixes.

        Raises
        ------
        ValueError
            If hyperparameter configuration is invalid.
        """
        # Check metrics configuration
        if hasattr(self, "_logging_metrics"):
            valid_metrics = {"MSE", "MAE", "MAPE", "SMAPE", "RMSE"}
            invalid_metrics = set(self._logging_metrics) - valid_metrics
            if invalid_metrics:
                msg = (
                    f"Invalid metric names in logging_metrics: {invalid_metrics}. "
                    f"Valid metrics are: {valid_metrics}"
                )
                raise ValueError(msg)

        # Check compilation compatibility
        if self.hparams.get("compile_model") and self.hparams.get("log_grad_norm"):
            log.warning(
                "Model compilation with gradient norm logging may cause performance issues. "
                "Consider disabling log_grad_norm for better compilation performance."
            )

    def _validate_batch_structure(
        self,
        batch: TimeSeriesSample | EncoderDecoderTargetSample,
        stage: str = "training",
    ) -> None:
        """
        Validate the structure and types of input batch.

        Parameters
        ----------
        batch : TimeSeriesSample or EncoderDecoderTargetSample
            The input batch to validate.
        stage : str, default="training"
            The current stage (training, validation, test, predict) for error messages.

        Raises
        ------
        TypeError
            If batch is not a dictionary or contains invalid types.
        ValueError
            If required keys are missing or tensor shapes are invalid.
        """
        if not isinstance(batch, dict):
            msg = (
                f"Expected batch to be a dictionary in {stage} stage, got {type(batch)}"
            )
            raise TypeError(msg)

        # Check required keys
        if "target" in batch:
            # Training/validation batch should have target
            target = batch["target"]
            if not isinstance(target, torch.Tensor):
                msg = f"Expected 'target' to be torch.Tensor in {stage} stage, got {type(target)}"
                raise TypeError(msg)

            if target.dim() < 2:
                msg = (
                    f"Expected 'target' to have at least 2 dimensions in {stage} stage "
                    f"(batch_size, seq_len), got shape {target.shape}"
                )
                raise ValueError(msg)

    def _validate_tensor_shapes(
        self, outputs: typing.Any, target: torch.Tensor, stage: str = "training"
    ) -> None:
        """
        Validate tensor shapes for metric calculation.

        Parameters
        ----------
        outputs : Any
            Model outputs to validate.
        target : torch.Tensor
            Target tensor to validate against.
        stage : str, default="training"
            Current stage for error messages.

        Raises
        ------
        TypeError
            If outputs or target have invalid types.
        ValueError
            If tensor shapes are incompatible.
        """
        if not isinstance(outputs, dict):
            msg = f"Expected outputs to be dict in {stage} stage, got {type(outputs)}"
            raise TypeError(msg)

        if "output" not in outputs and "point_prediction" not in outputs:
            msg = (
                f"Expected outputs to contain 'output' or 'point_prediction' key in {stage} stage. "
                f"Got keys: {list(outputs.keys())}"
            )
            raise ValueError(msg)

        # Get prediction tensor
        if "point_prediction" in outputs:
            prediction = outputs["point_prediction"]
        else:
            prediction = outputs["output"]

        if not isinstance(prediction, torch.Tensor):
            msg = f"Expected prediction to be torch.Tensor in {stage} stage, got {type(prediction)}"
            raise TypeError(msg)

        if not isinstance(target, torch.Tensor):
            msg = f"Expected target to be torch.Tensor in {stage} stage, got {type(target)}"
            raise TypeError(msg)

        # After squeezing, shapes should be compatible
        pred_squeezed = prediction.squeeze()
        target_squeezed = target.squeeze()

        if pred_squeezed.shape != target_squeezed.shape:
            msg = (
                f"Shape mismatch in {stage} stage: prediction {pred_squeezed.shape} "
                f"vs target {target_squeezed.shape} (after squeezing)"
            )
            raise ValueError(msg)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Log gradient norms before optimizer step if enabled in hyperparameters.

        This method calculates and logs the L2 norm of gradients for all model parameters
        when the "log_grad_norm" hyperparameter is set to True. This is useful for
        monitoring gradient flow and detecting issues like vanishing or exploding gradients.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer that will be used to update the model parameters.

        Notes
        -----
        - Only logs on global rank 0 to avoid duplicate logging in distributed training
        - Uses L2 norm (norm_type=2) for gradient norm calculation
        - Requires "log_grad_norm" hyperparameter to be set to True
        - Helpful for debugging training instability and gradient flow issues

        Examples
        --------
        >>> model = SomeModel(log_grad_norm=True)
        >>> # Gradient norms will be automatically logged during training
        >>> trainer.fit(model, datamodule)
        """
        if self.hparams.get("log_grad_norm") and self.global_rank == 0:
            self.log_dict(lightning.pytorch.utilities.grad_norm(self, norm_type=2))

    @property
    def validation_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        """
        Returns the validation outputs, which are stored in a dictionary where the keys are the dataloader indices and
        the values are lists of model outputs at each validation step.

        Returns
        -------
        dict[int, list[MODEL_OUTPUT]]
            Validation outputs, where the keys are the dataloader indices and the values are lists of model outputs.

        """
        return self._val_outputs

    @property
    def test_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        """
        Returns the test outputs, which are stored in a dictionary where the keys are the dataloader indices and
        the values are lists of model outputs at each test step.

        Returns
        -------
        dict[int, list[MODEL_OUTPUT]]
            Test outputs, where the keys are the dataloader indices and the values are lists of model outputs.
        """
        return self._test_outputs

    @property
    def inference_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        """
        Returns the inference outputs, which are stored in a dictionary where the keys are the dataloader indices and
        the values are lists of model outputs at each inference step.

        Returns
        -------
        dict[int, list[MODEL_OUTPUT]]
            Inference outputs, where the keys are the dataloader indices and the values are lists of model outputs.
        """
        return self._inference_outputs

    def state_dict(
        self,
        *args: typing.Any,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Return the state dictionary with proper handling of compiled models.

        This method extends the standard PyTorch state_dict to handle models that have
        been compiled with `torch.compile()`. When a model is compiled, PyTorch adds
        "_orig_mod" to parameter names, which this method removes to maintain
        compatibility with non-compiled model loading.

        Parameters
        ----------
        *args : Any
            Additional positional arguments passed to the parent state_dict method.
        destination : dict[str, torch.Tensor] or None, default=None
            If provided, the state dict will be copied into this dictionary.
        prefix : str, default=""
            Prefix to add to parameter names in the state dict.
        keep_vars : bool, default=False
            If True, returns Variables instead of Tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            The state dictionary with "_orig_mod" removed from parameter names
            if the model was compiled.

        Notes
        -----
        This method ensures that models saved after compilation can be loaded
        into non-compiled models and vice versa. The "_orig_mod" prefix is
        automatically added by PyTorch when using `torch.compile()`.

        Examples
        --------
        >>> model = SomeModel(compile_model=True)
        >>> trainer.fit(model, datamodule)
        >>> state_dict = model.state_dict()
        >>> # state_dict keys are normalized (no "_orig_mod" prefix)
        >>> torch.save(state_dict, "model.pth")
        """
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # hack to save the original model state dict and not the compiled one
        # this assumes that internally the model is stored in the `model` attribute
        # and that the model is not compiled when the LightningModule is instantiated

        # keys are xxx._orig_mod.xxx, remove _orig_mod
        if self.hparams.get("compile_model"):
            odict = collections.OrderedDict()
            for k in list(state_dict.keys()):
                if "_orig_mod" in k:
                    new_key = k.replace("_orig_mod.", "")
                    odict[new_key] = state_dict[k]
                else:
                    odict[k] = state_dict[k]

            state_dict = odict

        return state_dict
