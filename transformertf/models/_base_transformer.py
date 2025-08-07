from __future__ import annotations

import fnmatch
import sys
import typing
import warnings

import torch

from ..data import EncoderDecoderSample, EncoderDecoderTargetSample
from ..nn import QuantileLoss
from ..utils import ops
from ._base_module import LightningModuleBase

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class TransformerModuleBase(LightningModuleBase):
    """
    Base class for transformer-based time series forecasting models.

    This class extends :class:`LightningModuleBase` with transformer-specific functionality
    including encoder-decoder architectures, attention mechanisms, and specialized handling
    for quantile prediction. It provides standardized training, validation, and prediction
    workflows for transformer models.

    The class is designed to work with :class:`transformertf.data.EncoderDecoderTargetSample`
    and :class:`transformertf.data.EncoderDecoderSample` data structures, supporting both
    point predictions and probabilistic forecasting via quantile regression.

    Attributes
    ----------
    criterion : torch.nn.Module
        The loss function used for training. Supports both standard losses and
        :class:`transformertf.nn.QuantileLoss` for probabilistic forecasting.
    hparams : dict
        Hyperparameters including:
        - `compile_model`: Enable PyTorch compilation with dynamic shapes
        - `trainable_parameters`: List of parameter names to train (for transfer learning)
        - `prediction_type`: "point" or "delta" for different prediction modes

    Examples
    --------
    >>> from transformertf.models import TransformerModuleBase
    >>> from transformertf.data import EncoderDecoderDataModule
    >>> import torch
    >>>
    >>> class CustomTransformer(TransformerModuleBase):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.model = MyTransformerModel()
    ...         self.criterion = torch.nn.MSELoss()
    ...
    ...     def forward(self, batch):
    ...         return {"output": self.model(batch)}
    >>>
    >>> # Usage with encoder-decoder data
    >>> model = CustomTransformer(compile_model=True)
    >>> datamodule = EncoderDecoderDataModule(...)
    >>> trainer = L.Trainer()
    >>> trainer.fit(model, datamodule)

    Notes
    -----
    Key features of this base class:

    1. **Encoder-Decoder Architecture**: Designed for sequence-to-sequence tasks with
       separate encoder and decoder inputs.

    2. **Dynamic Compilation**: Supports PyTorch compilation with dynamic shapes,
       allowing for variable-length sequences while maintaining performance benefits.

    3. **Quantile Support**: Automatic handling of quantile losses for probabilistic
       forecasting, extracting point predictions from quantile outputs.

    4. **Transfer Learning**: Support for training only specific parameters via the
       `trainable_parameters` hyperparameter.

    5. **Prediction Types**:
       - "point": Direct target prediction
       - "delta": Predicts differences between consecutive time steps

    The class automatically handles the complexities of transformer training including
    attention masking, dynamic shape compilation, and quantile loss processing.

    See Also
    --------
    LightningModuleBase : Parent base class for all models
    transformertf.data.EncoderDecoderDataModule : Compatible data module
    transformertf.nn.QuantileLoss : Quantile loss for probabilistic forecasting
    get_attention_mask : Function for creating attention masks
    """

    criterion: torch.nn.Module

    @override
    def maybe_compile_model(self) -> None:
        """
        Compile the model with dynamic shape support for transformer architectures.

        This method overrides the base implementation to enable dynamic compilation
        specifically for transformer models. Unlike the base class, this version
        uses `dynamic=True` to handle variable-length sequences common in transformer
        architectures.

        The compilation applies to all child modules except those with "loss" in their
        name, allowing for optimized execution while maintaining loss function compatibility.

        Notes
        -----
        - Uses `dynamic=True` for handling variable sequence lengths
        - Particularly beneficial for transformer attention mechanisms
        - Requires PyTorch 2.0+ and compatible hardware for optimal performance
        - Works with the `_maybe_mark_dynamic` method for dynamic shape handling

        Examples
        --------
        >>> model = SomeTransformerModel(compile_model=True)
        >>> # Dynamic compilation is automatically applied during fit start
        >>> trainer.fit(model, datamodule)
        """
        if self.hparams.get("compile_model"):
            for name, mod in self.named_children():
                if "loss" in name.lower():
                    continue
                setattr(self, name, torch.compile(mod, dynamic=True))

    @override
    def on_fit_start(self) -> None:
        """
        Initialize transformer model for training with validation.
        """
        self._validate_transformer_hyperparameters()
        super().on_fit_start()

    def _validate_encoder_decoder_batch(
        self, batch: EncoderDecoderTargetSample, stage: str = "training"
    ) -> None:
        """
        Validate encoder-decoder batch structure and tensor shapes.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            The encoder-decoder batch to validate.
        stage : str, default="training"
            Current stage for error messages.

        Raises
        ------
        ValueError
            If required keys are missing or tensor shapes are invalid.
        TypeError
            If batch contains invalid types.
        """
        # Validate basic structure first
        self._validate_batch_structure(batch, stage)

        # Check encoder-decoder specific keys
        required_keys = ["encoder_input", "decoder_input"]
        for key in required_keys:
            if key not in batch:
                msg = f"Expected '{key}' key in batch during {stage}, got keys: {list(batch.keys())}"
                raise ValueError(msg)

            tensor = batch[key]
            if not isinstance(tensor, torch.Tensor):
                msg = f"Expected '{key}' to be torch.Tensor in {stage}, got {type(tensor)}"
                raise TypeError(msg)

            if tensor.dim() != 3:
                msg = (
                    f"Expected '{key}' to have 3 dimensions (batch_size, seq_len, features) "
                    f"in {stage}, got shape {tensor.shape}"
                )
                raise ValueError(msg)

        # Check sequence length keys if present
        for length_key in ["encoder_lengths", "decoder_lengths"]:
            if length_key in batch:
                lengths = batch[length_key]
                if not isinstance(lengths, torch.Tensor):
                    msg = f"Expected '{length_key}' to be torch.Tensor in {stage}, got {type(lengths)}"
                    raise TypeError(msg)

                if lengths.dim() != 2 or lengths.shape[1] != 1:
                    msg = (
                        f"Expected '{length_key}' to have shape (batch_size, 1) in {stage}, "
                        f"got shape {lengths.shape}"
                    )
                    raise ValueError(msg)

    def _extract_point_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Extract point prediction from model output, handling quantile models.

        For quantile models, this extracts the median (point prediction).
        For regular models, returns the output as-is.

        Parameters
        ----------
        model_output : torch.Tensor
            Raw model output tensor.

        Returns
        -------
        torch.Tensor
            Point prediction tensor.
        """
        if isinstance(self.criterion, QuantileLoss) or (
            hasattr(self.criterion, "_orig_mod")
            and isinstance(self.criterion._orig_mod, QuantileLoss)  # noqa: SLF001
        ):
            return self.criterion.point_prediction(model_output)
        return model_output

    def _prepare_step_output(
        self, model_output: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Prepare the output dictionary for training/validation steps.

        This method handles point prediction extraction, output detachment,
        and result dictionary construction.

        Parameters
        ----------
        model_output : dict[str, torch.Tensor]
            Model output dictionary containing "output" and other keys.
        loss : torch.Tensor
            Computed loss value.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing loss, output, point_prediction, and other outputs.
        """
        loss_dict = {"loss": loss}
        point_prediction = self._extract_point_prediction(model_output["output"])

        return {
            **loss_dict,
            "output": model_output.pop("output"),
            **{k: ops.detach(v) for k, v in model_output.items()},
            "point_prediction": point_prediction,
        }

    def on_train_batch_start(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> None:
        self._maybe_mark_dynamic(batch)

        super().on_train_batch_start(batch=batch, batch_idx=batch_idx)

    def on_validation_batch_start(
        self, batch: EncoderDecoderTargetSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._maybe_mark_dynamic(batch)

        super().on_validation_batch_start(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single training step for transformer-based models.

        This method processes a batch of encoder-decoder samples, computes the model
        output, calculates the loss, and returns the results with proper handling
        of quantile predictions.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            A batch containing encoder inputs, decoder inputs, and target values.
            Expected keys: "encoder_input", "decoder_input", "target".
        batch_idx : int
            Index of the current batch within the epoch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": The computed loss value
            - "output": The raw model output
            - "point_prediction": Point estimate (median for quantile models)
            - Additional model outputs (detached for memory efficiency)

        Raises
        ------
        ValueError
            If required keys are missing or tensor shapes are invalid.
        TypeError
            If batch contains invalid types.

        Notes
        -----
        This method automatically handles:
        - Input validation with helpful error messages
        - Quantile loss point prediction extraction
        - Proper detachment of auxiliary outputs to prevent memory leaks
        - Logging of training metrics via `common_log_step`
        - Support for both compiled and non-compiled quantile losses

        The method works with both standard losses and quantile losses, automatically
        extracting point predictions from quantile outputs when applicable.
        """
        # Validate encoder-decoder batch structure
        self._validate_encoder_decoder_batch(batch, "training")

        model_output = self(batch)
        loss = self.calc_loss(model_output["output"], batch)

        # Log training metrics
        self.common_log_step({"loss": loss}, "train")

        # Prepare and return step output
        return self._prepare_step_output(model_output, loss)

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single validation step for transformer-based models.

        This method processes a validation batch, computes the model output and loss,
        and returns the results. Similar to training_step but without gradient computation.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            A validation batch containing encoder inputs, decoder inputs, and target values.
            Expected keys: "encoder_input", "decoder_input", "target".
        batch_idx : int
            Index of the current batch within the validation epoch.
        dataloader_idx : int, default=0
            Index of the dataloader when multiple validation dataloaders are used.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": The computed validation loss value
            - "output": The raw model output
            - "point_prediction": Point estimate (median for quantile models)
            - Additional model outputs (detached for memory efficiency)

        Raises
        ------
        ValueError
            If required keys are missing or tensor shapes are invalid.
        TypeError
            If batch contains invalid types.

        Notes
        -----
        This method:
        - Validates input batch structure with helpful error messages
        - Runs in no-grad mode (handled by Lightning)
        - Automatically handles quantile loss point prediction extraction
        - Logs validation metrics via `common_log_step`
        - Supports multiple validation dataloaders
        - Output collection is handled by the parent class for later analysis

        The validation outputs are automatically collected by the base class and
        can be accessed via the `validation_outputs` property.
        """
        # Validate encoder-decoder batch structure
        self._validate_encoder_decoder_batch(batch, "validation")

        model_output = self(batch)
        loss = self.calc_loss(model_output["output"], batch)

        # Log validation metrics
        self.common_log_step({"loss": loss}, "validation")

        # Prepare and return step output
        return self._prepare_step_output(model_output, loss)

    def test_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single test step for transformer-based models.

        This method processes a test batch, computes the model output and optionally
        the loss (if targets are available), and returns the results. Used for
        evaluation on test datasets and inference on new data.

        Parameters
        ----------
        batch : EncoderDecoderTargetSample
            A test batch containing encoder inputs, decoder inputs, and optionally target values.
            Expected keys: "encoder_input", "decoder_input".
            Optional key: "target" (for evaluation with loss computation).
        batch_idx : int
            Index of the current batch within the test epoch.
        dataloader_idx : int, default=0
            Index of the dataloader when multiple test dataloaders are used.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": The computed test loss value (only if target is available)
            - "output": The raw model output
            - "point_prediction": Point estimate (median for quantile models)
            - Additional model outputs (detached for memory efficiency)

        Raises
        ------
        ValueError
            If required keys are missing or tensor shapes are invalid.
        TypeError
            If batch contains invalid types.

        Notes
        -----
        This method:
        - Validates input batch structure with helpful error messages
        - Runs in no-grad mode (handled by Lightning)
        - Conditionally computes loss only if target is available in batch
        - Automatically handles quantile loss point prediction extraction
        - Logs test metrics via `common_log_step` (only when loss is computed)
        - Supports multiple test dataloaders
        - Handles both evaluation scenarios (with targets) and inference scenarios (without targets)

        The test outputs are automatically collected by the base class and
        can be accessed via the `test_outputs` property.

        Examples
        --------
        >>> # Test with targets for evaluation
        >>> batch_with_target = {
        ...     "encoder_input": encoder_data,
        ...     "decoder_input": decoder_data,
        ...     "target": target_data
        ... }
        >>> output = model.test_step(batch_with_target, 0)
        >>> print(f"Test loss: {output['loss']}")
        >>>
        >>> # Test without targets for inference
        >>> batch_inference = {
        ...     "encoder_input": encoder_data,
        ...     "decoder_input": decoder_data
        ... }
        >>> output = model.test_step(batch_inference, 0)
        >>> print(f"Predictions: {output['point_prediction']}")
        """
        # Validate encoder-decoder batch structure
        self._validate_encoder_decoder_batch(batch, "test")

        model_output = self(batch)

        # Conditionally compute loss if target is available
        if "target" in batch:
            loss = self.calc_loss(model_output["output"], batch)

            # Log test metrics
            self.common_log_step({"loss": loss}, "test")

            # Prepare and return step output with loss
            return self._prepare_step_output(model_output, loss)
        # No target available - return model output without loss
        point_prediction = self._extract_point_prediction(model_output["output"])

        return {
            "output": model_output.pop("output"),
            **{k: ops.detach(v) for k, v in model_output.items()},
            "point_prediction": point_prediction,
        }

    def predict_step(
        self, batch: EncoderDecoderSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single prediction step for transformer-based models.

        This method processes a prediction batch (without targets) and returns the
        model output with point predictions. Used for inference and generating
        forecasts on new data.

        Parameters
        ----------
        batch : EncoderDecoderSample
            A prediction batch containing encoder inputs and decoder inputs.
            Expected keys: "encoder_input", "decoder_input".
            Note: No "target" key is expected during prediction.
        batch_idx : int
            Index of the current batch within the prediction run.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "output": The raw model output
            - "point_prediction": Point estimate (median for quantile models)
            - Additional model outputs (e.g., attention weights, hidden states)

        Notes
        -----
        This method:
        - Runs in no-grad mode (handled by Lightning)
        - Does not compute loss (no targets available)
        - Automatically extracts point predictions from quantile outputs
        - Returns all model outputs for comprehensive analysis
        - Outputs are collected by the base class via `inference_outputs` property

        For quantile models, the point prediction corresponds to the median
        quantile, providing the best single-point estimate.

        Examples
        --------
        >>> # During prediction
        >>> predictions = trainer.predict(model, dataloader)
        >>> # Access collected outputs
        >>> all_outputs = model.inference_outputs
        """
        model_output = self(batch)

        # Add point prediction to model output
        point_prediction = self._extract_point_prediction(model_output["output"])
        model_output["point_prediction"] = point_prediction

        return model_output

    def _validate_transformer_hyperparameters(self) -> None:
        """
        Validate transformer-specific hyperparameter configuration.

        Raises
        ------
        ValueError
            If transformer hyperparameter configuration is invalid.
        """
        # Validate prediction type and issue deprecation warning
        prediction_type = self.hparams.get("prediction_type")
        if prediction_type is not None:
            warnings.warn(
                "The 'prediction_type' parameter is deprecated. To predict deltas, "
                "use DeltaTransform in the data preprocessing pipeline by adding "
                "it to data.init_args.extra_transforms in your config: "
                "{'target_covariate_name': [{'class_path': 'transformertf.data.transform.DeltaTransform'}]}",
                DeprecationWarning,
                stacklevel=3,
            )

            # Maintain existing validation for backward compatibility
            if prediction_type not in {"point", "delta"}:
                msg = (
                    f"Invalid prediction_type '{prediction_type}'. "
                    f"Must be 'point' or 'delta'."
                )
                raise ValueError(msg)

        # Validate trainable parameters
        trainable_params = self.hparams.get("trainable_parameters")
        if trainable_params is not None:
            if isinstance(trainable_params, list):
                # Validate list format (backward compatible)
                if not all(isinstance(param, str) for param in trainable_params):
                    msg = "All items in trainable_parameters list must be strings"
                    raise ValueError(msg)
            elif isinstance(trainable_params, dict):
                # Validate dict format (enhanced)
                valid_keys = {"include", "exclude"}
                if not set(trainable_params.keys()).issubset(valid_keys):
                    invalid_keys = set(trainable_params.keys()) - valid_keys
                    msg = f"Invalid keys in trainable_parameters dict: {invalid_keys}. Valid keys: {valid_keys}"
                    raise ValueError(msg)

                # Validate include patterns
                include_patterns = trainable_params.get("include", ["*"])
                if not isinstance(include_patterns, list) or not all(
                    isinstance(p, str) for p in include_patterns
                ):
                    msg = "trainable_parameters['include'] must be a list of strings"
                    raise ValueError(msg)

                # Validate exclude patterns
                exclude_patterns = trainable_params.get("exclude", [])
                if not isinstance(exclude_patterns, list) or not all(
                    isinstance(p, str) for p in exclude_patterns
                ):
                    msg = "trainable_parameters['exclude'] must be a list of strings"
                    raise ValueError(msg)
            else:
                msg = f"trainable_parameters must be a list, dict, or None, got {type(trainable_params)}"
                raise ValueError(msg)

    def on_train_epoch_start(self) -> None:
        """
        Set normalizing layers not in trainable_parameters to eval mode.

        This ensures that LayerNorm layers that are not being trained maintain
        their statistics from pretraining, which is important for transfer learning.
        """
        if (
            "trainable_parameters" not in self.hparams
            or self.hparams["trainable_parameters"] is None
        ):
            return

        config = self.hparams["trainable_parameters"]

        # Handle both list and dict formats
        if isinstance(config, list):
            include_patterns = config
            exclude_patterns = []
        else:
            include_patterns = config.get("include", ["*"])
            exclude_patterns = config.get("exclude", [])

        for name, module in self.named_modules():
            if not isinstance(module, torch.nn.LayerNorm):
                continue

            # Remove "model." prefix if present to get clean component path
            clean_name = name[6:] if name.startswith("model.") else name

            # Check if this module should be trainable using same logic as parameters()
            included = any(
                self._matches_pattern(clean_name, pattern)
                for pattern in include_patterns
            )
            excluded = any(
                self._matches_pattern(clean_name, pattern)
                for pattern in exclude_patterns
            )

            # Set to eval mode if not trainable
            if not (included and not excluded):
                module.eval()

    def parameters(self, recurse: bool = True) -> typing.Iterator[torch.nn.Parameter]:
        """
        Override the parameters method to only return the trainable parameters.

        This method uses parameter filtering rather than requires_grad=False because
        transformer fine-tuning often requires freezing output/head layers. Setting
        requires_grad=False on final layers would break gradient flow to all upstream
        parameters, preventing training of earlier layers. Parameter filtering allows
        selective optimization while maintaining full gradient computation throughout
        the network.

        Parameters
        ----------
        recurse : bool, optional
            Whether to return parameters of this module and all submodules,
            by default True

        Returns
        -------
        Iterator[torch.nn.Parameter]
            Iterator over trainable parameters based on trainable_parameters configuration

        Notes
        -----
        The trainable_parameters configuration supports two formats:

        1. **Simple list format** (backward compatible):
           trainable_parameters = ["encoder", "decoder.attention"]

        2. **Include/exclude dict format** (enhanced):
           trainable_parameters = {
               "include": ["encoder.*", "decoder.attention.*"],
               "exclude": ["encoder.layer.0.*", "*.bias"]
           }

        Pattern matching uses glob syntax:
        - "*" matches any sequence of characters
        - "?" matches any single character
        - "[seq]" matches any character in seq
        - "encoder.*" matches all parameters in encoder module
        - "*.weight" matches all weight parameters
        - "encoder.layer.[0-2].*" matches first 3 encoder layers

        Examples
        --------
        >>> # Freeze embeddings and first encoder layer, train rest
        >>> trainable_parameters = {
        ...     "include": ["*"],
        ...     "exclude": ["embeddings.*", "encoder.layer.0.*"]
        ... }
        >>>
        >>> # Train only attention and output layers
        >>> trainable_parameters = ["encoder.attention", "decoder.attention", "head"]
        >>>
        >>> # Fine-tune only the last few transformer layers
        >>> trainable_parameters = {
        ...     "include": ["encoder.layer.[6-11].*", "head.*"]
        ... }
        """
        if self.hparams["trainable_parameters"] is None:
            yield from super().parameters(recurse=recurse)
            return

        config = self.hparams["trainable_parameters"]

        # Handle both list and dict formats for backward compatibility
        if isinstance(config, list):
            include_patterns = config
            exclude_patterns = []
        else:
            include_patterns = config.get("include", ["*"])
            exclude_patterns = config.get("exclude", [])

        for name, param in self.named_parameters(recurse=recurse):
            # Remove "model." prefix if present to get clean component path
            clean_name = name[6:] if name.startswith("model.") else name

            # Check inclusion patterns
            included = any(
                self._matches_pattern(clean_name, pattern)
                for pattern in include_patterns
            )

            # Check exclusion patterns
            excluded = any(
                self._matches_pattern(clean_name, pattern)
                for pattern in exclude_patterns
            )

            if included and not excluded:
                yield param

    def _matches_pattern(self, param_name: str, pattern: str) -> bool:
        """
        Check if a parameter name matches a glob pattern.

        Supports both full path matching and legacy component matching for
        backward compatibility with existing trainable_parameters configurations.

        Parameters
        ----------
        param_name : str
            The parameter name to check (e.g., "encoder.layer.0.weight")
        pattern : str
            The glob pattern to match against (e.g., "encoder.layer.*")

        Returns
        -------
        bool
            True if the parameter name matches the pattern

        Examples
        --------
        >>> self._matches_pattern("encoder.layer.0.weight", "encoder.*")
        True
        >>> self._matches_pattern("decoder.attention.weight", "encoder.*")
        False
        >>> self._matches_pattern("head.classifier.bias", "*.bias")
        True
        >>> # Legacy compatibility - matches component name
        >>> self._matches_pattern("enc_vs.single_grn.0.weight", "enc_vs")
        True
        """
        # First try full path matching (new enhanced format)
        if fnmatch.fnmatch(param_name, pattern):
            return True

        # For backward compatibility, also try matching against the top-level component
        # This maintains compatibility with old trainable_parameters lists
        if "." in param_name:
            component_name = param_name.split(".", 1)[0]
            if component_name == pattern or fnmatch.fnmatch(component_name, pattern):
                return True

        return False

    def calc_loss(
        self,
        model_output: torch.Tensor,
        batch: EncoderDecoderTargetSample,
    ) -> torch.Tensor:
        """
        Calculate the loss based on model output and target values.

        This method computes the loss for point prediction (absolute values).
        For delta prediction (differences), use DeltaTransform in the data
        preprocessing pipeline instead.

        Parameters
        ----------
        model_output : torch.Tensor
            The raw output from the model, typically of shape (batch_size, seq_len, features).
        batch : EncoderDecoderTargetSample
            The batch containing target values.
            Expected keys: "target".

        Returns
        -------
        torch.Tensor
            The computed loss value.

        Examples
        --------
        >>> # Point prediction
        >>> loss = self.calc_loss(model_output, batch)
        """
        weights = None
        target = batch["target"]

        # Squeeze target only if model output has been squeezed (for compatibility)
        if model_output.dim() == target.dim() - 1:
            target = target.squeeze(-1)

        # Pass weights only if criterion supports it (e.g., QuantileLoss)
        if (
            hasattr(self.criterion, "forward")
            and "weights" in self.criterion.forward.__code__.co_varnames
        ):
            return typing.cast(
                torch.Tensor, self.criterion(model_output, target, weights=weights)
            )
        return typing.cast(torch.Tensor, self.criterion(model_output, target))

    def _maybe_mark_dynamic(
        self, batch: EncoderDecoderTargetSample
    ) -> EncoderDecoderTargetSample:
        """
        Mark the input tensors as dynamic so that the torch compiler can optimize the
        computation graph, even when input shapes are changing.

        Needs PyTorch 2.7.0 to work when distributed training (DDP) is enabled.
        https://github.com/pytorch/pytorch/issues/140229
        """
        if not self.hparams["compile_model"]:
            return batch

        torch._dynamo.mark_dynamic(batch["encoder_input"], index=1)  # noqa: SLF001
        torch._dynamo.mark_dynamic(batch["decoder_input"], index=1)  # noqa: SLF001
        if "target" in batch:
            torch._dynamo.mark_dynamic(batch["target"], index=1)  # noqa: SLF001

        return batch


def get_attention_mask(
    encoder_lengths: torch.LongTensor,
    decoder_lengths: torch.LongTensor,
    max_encoder_length: int,
    max_decoder_length: int,
    *,
    causal_attention: bool = True,
    encoder_alignment: str = "left",
    decoder_alignment: str = "left",
) -> torch.Tensor:
    """
    Create attention masks for transformer encoder-decoder architectures.

    This function generates attention masks that handle variable-length sequences
    and implement causal masking for autoregressive generation. The mask ensures
    that the decoder can attend to all encoder positions and appropriate decoder
    positions based on the causality constraint.

    Parameters
    ----------
    encoder_lengths : torch.LongTensor
        Actual lengths of encoder sequences in the batch, shape (batch_size,).
        Used to mask padding tokens in encoder sequences.
    decoder_lengths : torch.LongTensor
        Actual lengths of decoder sequences in the batch, shape (batch_size,).
        Used to mask padding tokens in decoder sequences.
    max_encoder_length : int
        Maximum length of encoder sequences in the batch.
    max_decoder_length : int
        Maximum length of decoder sequences in the batch.
    causal_attention : bool, default=True
        If True, apply causal masking where each decoder position can only
        attend to previous positions. If False, allows attention to all
        non-padded positions.
    encoder_alignment : str, default="left"
        Alignment of encoder sequences. Either "left" (padding at start) or "right" (padding at end).
    decoder_alignment : str, default="left"
        Alignment of decoder sequences. Either "left" (padding at start) or "right" (padding at end).

    Returns
    -------
    torch.Tensor
        Attention mask of shape (batch_size, max_decoder_length, max_encoder_length + max_decoder_length).
        True indicates positions that should be masked (not attended to).

    Notes
    -----
    The returned mask combines:

    1. **Encoder mask**: Masks padded positions in encoder sequences
    2. **Decoder mask**: Masks padded positions and implements causal constraints

    For causal attention (default):
    - Decoder positions can attend to all encoder positions
    - Decoder position i can only attend to decoder positions 0 to i-1

    For non-causal attention:
    - Decoder positions can attend to all encoder positions
    - Decoder positions can attend to all non-padded decoder positions

    Examples
    --------
    >>> encoder_lengths = torch.tensor([10, 8, 12])
    >>> decoder_lengths = torch.tensor([5, 7, 6])
    >>> mask = get_attention_mask(
    ...     encoder_lengths, decoder_lengths,
    ...     max_encoder_length=12, max_decoder_length=7,
    ...     causal_attention=True
    ... )
    >>> # mask.shape = (3, 7, 19)  # batch_size=3, dec_len=7, enc_len+dec_len=19

    See Also
    --------
    create_mask : Utility function for creating basic padding masks
    """
    if causal_attention:
        # indices to which is attended
        attend_step = torch.arange(max_decoder_length, device=encoder_lengths.device)
        # indices for which is predicted
        predict_step = torch.arange(
            0, max_decoder_length, device=encoder_lengths.device
        )[:, None]
        # do not attend to steps to self or after prediction
        decoder_mask = (
            (attend_step >= predict_step)
            .unsqueeze(0)
            .expand(encoder_lengths.size(0), -1, -1)
        )
    else:
        # there is value in attending to future forecasts if
        # they are made with knowledge currently available
        #   one possibility is here to use a second attention layer
        # for future attention
        # (assuming different effects matter in the future than the past)
        #  or alternatively using the same layer but
        # allowing forward attention - i.e. only
        #  masking out non-available data and self
        decoder_mask = (
            create_mask(
                max_decoder_length, decoder_lengths, alignment=decoder_alignment
            )
            .unsqueeze(1)
            .expand(-1, max_decoder_length, -1)
        )
    # do not attend to steps where data is padded
    encoder_mask = (
        create_mask(max_encoder_length, encoder_lengths, alignment=encoder_alignment)
        .unsqueeze(1)
        .expand(-1, max_decoder_length, -1)
    )
    # combine masks along attended time - first encoder and then decoder
    return torch.cat(
        (
            encoder_mask,
            decoder_mask,
        ),
        dim=2,
    )


def create_mask(
    size: int,
    lengths: torch.LongTensor,
    *,
    alignment: str = "left",
    inverse: bool = False,
) -> torch.BoolTensor:
    """
    Create boolean masks for variable-length sequences.

    This utility function generates boolean masks that indicate valid positions
    in padded sequences. It's commonly used in transformer models to mask
    padding tokens in attention computations.

    Parameters
    ----------
    size : int
        The maximum sequence length (size of second dimension).
    lengths : torch.LongTensor
        Tensor of actual sequence lengths for each item in the batch.
        Shape: (batch_size,).
    alignment : str, default="left"
        Sequence alignment type:
        - "left": Sequences are left-aligned (padding at start)
        - "right": Sequences are right-aligned (padding at end)
    inverse : bool, default=False
        If False, returns True where positions are invalid (padded).
        If True, returns True where positions are valid (not padded).

    Returns
    -------
    torch.BoolTensor
        Boolean mask of shape (len(lengths), size).

        For left alignment:
        - When inverse=False: mask[i, j] = True if j < (size - lengths[i]) (padded position)
        - When inverse=True: mask[i, j] = True if j >= (size - lengths[i]) (valid position)

        For right alignment:
        - When inverse=False: mask[i, j] = True if lengths[i] <= j (padded position)
        - When inverse=True: mask[i, j] = True if lengths[i] > j (valid position)

    Examples
    --------
    >>> lengths = torch.tensor([3, 5, 2])
    >>>
    >>> # Right alignment (traditional behavior)
    >>> mask = create_mask(size=6, lengths=lengths, alignment="right", inverse=False)
    >>> # mask = [[False, False, False, True, True, True],     # len=3, padding at end
    >>> #          [False, False, False, False, False, True],  # len=5, padding at end
    >>> #          [False, False, True, True, True, True]]     # len=2, padding at end
    >>>
    >>> # Left alignment (new default)
    >>> mask = create_mask(size=6, lengths=lengths, alignment="left", inverse=False)
    >>> # mask = [[True, True, True, False, False, False],     # len=3, padding at start
    >>> #          [True, False, False, False, False, False],  # len=5, padding at start
    >>> #          [True, True, True, True, False, False]]     # len=2, padding at start

    Notes
    -----
    This function is commonly used in transformer attention mechanisms to:
    - Mask padding tokens so they don't participate in attention
    - Create causal masks for autoregressive generation
    - Handle variable-length sequences in batched operations

    The alignment parameter is critical for correct masking behavior:
    - Use "left" when sequences have been aligned for RNN packing (padding at start)
    - Use "right" for traditional padding (padding at end)

    See Also
    --------
    get_attention_mask : Higher-level function that uses this for attention masking
    """

    if alignment not in {"left", "right"}:
        msg = f"alignment must be 'left' or 'right', got '{alignment}'"
        raise ValueError(msg)

    indices = torch.arange(size, device=lengths.device).unsqueeze(0)

    if alignment == "right":
        # Right alignment: padding at end
        if inverse:  # return where values are
            return indices < lengths.unsqueeze(-1)
        # return where no values are (padding positions)
        return indices >= lengths.unsqueeze(-1)

    # Left alignment: padding at start
    padding_start_positions = size - lengths.unsqueeze(-1)
    if inverse:  # return where values are
        return indices >= padding_start_positions
    # return where no values are (padding positions)
    return indices < padding_start_positions
