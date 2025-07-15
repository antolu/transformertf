Extending the API with New Models
==================================

This guide demonstrates how to extend the TransformerTF API with new models, using the MLP hysteresis model from the ``mlp.ipynb`` notebook as a practical example. This tutorial is designed for newcomers to the framework and explains each step in detail.

What You'll Learn
-----------------

By the end of this guide, you'll understand:

- How to create a new model that works with TransformerTF
- The structure and requirements for custom models
- How to train and evaluate your custom model
- Best practices for model development

Understanding the Framework
---------------------------

TransformerTF is built on top of PyTorch Lightning, which provides a structured way to organize machine learning code. Every model in TransformerTF inherits from a base class called ``LightningModuleBase``. This base class handles many common tasks automatically, such as:

- Saving and loading model parameters (hyperparameters)
- Calculating evaluation metrics like MSE, MAE, MAPE, SMAPE, RMSE
- Logging training progress
- Managing model compilation for better performance

The Architecture Pattern
-------------------------

Basic Model Structure
~~~~~~~~~~~~~~~~~~~~~

Every model in TransformerTF follows the same basic pattern. Here's what a typical model looks like:

.. code-block:: python

    from __future__ import annotations

    import torch
    import lightning as L
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from transformertf.models import LightningModuleBase
    from transformertf.data import EncoderDecoderTargetSample
    from transformertf.nn import MLP, GatedResidualNetwork

    class MyNewModel(LightningModuleBase):
        def __init__(
            self,
            # Your model's specific parameters go here
            param1: int,
            param2: float = 0.1,
            # Standard parameters that most models need
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            *,
            # Framework-specific parameters
            compile_model: bool = False,
        ):
            # This calls the parent class constructor
            super().__init__()
            # This saves all the parameters so Lightning can restore them later
            self.save_hyperparameters()

            # Here you create your neural network layers
            # Here you set up your loss function

        def configure_optimizers(self) -> OptimizerLRScheduler:
            # This method tells Lightning how to optimize your model
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # This is where your model processes input data
            pass

        def common_step(self, batch: dict) -> dict:
            # This contains logic shared by training, validation, and testing
            pass

        def training_step(self, batch: dict, batch_idx: int) -> dict:
            # This runs for each batch during training
            pass

        def validation_step(self, batch: dict, batch_idx: int) -> dict:
            # This runs for each batch during validation
            pass

Let's break down what each part does:

1. **__init__**: This is where you define your model's parameters and create the neural network layers
2. **configure_optimizers**: This tells Lightning how to update your model's weights during training
3. **forward**: This is the main computation of your model - it takes input data and produces predictions
4. **common_step**: This contains logic that's the same for training, validation, and testing
5. **training_step**: This runs during training and typically just calls common_step
6. **validation_step**: This runs during validation and typically just calls common_step

Complete Example: MLP Hysteresis Model
--------------------------------------

Now let's look at a complete, working example. This model predicts magnetic field hysteresis - it takes three inputs (past magnetic field, current, and next current) and predicts the next magnetic field.

.. code-block:: python

    from __future__ import annotations

    import torch
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from transformertf.models import LightningModuleBase
    from transformertf.data import EncoderDecoderTargetSample
    from transformertf.nn import MLP, GatedResidualNetwork


    class MLPHysteresis(LightningModuleBase):
        """
        Multi-layer Perceptron (MLP) for hysteresis prediction.

        This model predicts magnetic field hysteresis using a combination of:
        - MLP (Multi-Layer Perceptron): A basic neural network with multiple layers
        - GRN (Gated Residual Network): A more sophisticated layer that can selectively
          pass information through "gates"

        The model takes 3 inputs:
        1. Past magnetic field value
        2. Current electrical current
        3. Next electrical current

        And outputs:
        1. Next magnetic field value
        """

        def __init__(
            self,
            num_layers: int,                    # How many hidden layers in the MLP
            hidden_size: int = 64,              # How many neurons in each layer
            dropout: float = 0.1,               # Dropout rate to prevent overfitting
            lr: float = 1e-3,                   # Learning rate for training
            weight_decay: float = 0.0,          # Weight decay for regularization
            *,
            compile_model: bool = False,        # Whether to use PyTorch 2.0 compilation
        ):
            super().__init__()
            self.save_hyperparameters()

            # Create the main MLP network
            # It takes 3 inputs and outputs 'hidden_size' features
            self.mlp = MLP(
                input_dim=3,  # past B, current I, next I
                output_dim=hidden_size,
                hidden_dim=[hidden_size] * num_layers,  # All layers have same size
                dropout=dropout,
                activation="relu",  # ReLU activation function
            )

            # Create two Gated Residual Networks
            # These help the model learn complex patterns
            self.grn1 = GatedResidualNetwork(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
            )

            self.grn2 = GatedResidualNetwork(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
            )

            # Final layer that outputs just 1 value (the predicted magnetic field)
            self.linear = torch.nn.Linear(hidden_size, 1)

            # Set up the loss function (how we measure prediction errors)
            # MSE = Mean Squared Error
            self.criterion = torch.nn.MSELoss()

        def configure_optimizers(self) -> OptimizerLRScheduler:
            """
            Configure the optimizer (the algorithm that updates model weights).

            We use AdamW, which is a popular and effective optimizer for neural networks.
            """
            optimizer = torch.optim.AdamW(
                self.parameters(),  # All the model's trainable parameters
                lr=self.hparams.lr,  # Learning rate (how big steps to take)
                weight_decay=self.hparams.weight_decay  # Regularization strength
            )
            return optimizer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass - this is where the actual computation happens.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor with shape (batch_size, sequence_length, 3)
                The 3 features are: [past_B, current_I, next_I]

            Returns
            -------
            torch.Tensor
                Output tensor with shape (batch_size, sequence_length, 1)
                Contains the predicted next magnetic field
            """
            # Step 1: Process input through the MLP
            x = self.mlp(x)

            # Step 2: Save the MLP output for the residual connection
            residual = x

            # Step 3: Apply the two GRNs
            x = self.grn1(x)
            x = self.grn2(x)

            # Step 4: Add the residual connection
            # This helps with training stability and gradient flow
            x = x + residual

            # Step 5: Final linear layer to get the prediction
            x = self.linear(x)

            return x

        def common_step(self, batch: EncoderDecoderTargetSample) -> dict:
            """
            Common logic for training, validation, and test steps.

            This function:
            1. Extracts the input features from the batch
            2. Runs the model forward pass
            3. Calculates the loss
            4. Returns results in a standard format

            Parameters
            ----------
            batch : EncoderDecoderTargetSample
                A dictionary containing:
                - "encoder_input": Past context data
                - "decoder_input": Current input data
                - "target": What we want to predict

            Returns
            -------
            dict
                Dictionary with "loss", "output", and "point_prediction"
            """
            # Extract the input features
            # encoder_input contains: [past_B, current_I] (shape: batch_size, 1, 2)
            # decoder_input contains: [next_I, target_B] (shape: batch_size, 1, 2)
            # We want to create: [past_B, current_I, next_I] (shape: batch_size, 1, 3)

            x = torch.cat([
                batch["encoder_input"],              # [past_B, current_I]
                batch["decoder_input"][..., :-1]    # [next_I] (exclude the target)
            ], dim=-1)

            # Get the target (what we want to predict)
            y = batch["target"]

            # Run the model forward pass to get predictions
            y_hat = self(x)

            # Calculate how wrong our predictions are
            loss = self.criterion(y_hat, y)

            # Return results in the expected format
            return {
                "loss": loss,                    # The loss value for training
                "output": y_hat,                 # The raw model output
                "point_prediction": y_hat,       # The prediction for evaluation metrics
            }

        def training_step(self, batch: EncoderDecoderTargetSample, batch_idx: int) -> dict:
            """
            Training step - called for each batch during training.
            """
            step_output = self.common_step(batch)
            # Log the loss so we can monitor training progress
            self.common_log_step({"loss": step_output["loss"]}, "train")
            return step_output

        def validation_step(self, batch: EncoderDecoderTargetSample, batch_idx: int) -> dict:
            """
            Validation step - called for each batch during validation.
            """
            step_output = self.common_step(batch)
            # Log the loss so we can monitor validation progress
            self.common_log_step({"loss": step_output["loss"]}, "validation")
            return step_output

        def test_step(self, batch: EncoderDecoderTargetSample, batch_idx: int) -> dict:
            """
            Test step - called for each batch during testing.
            """
            step_output = self.common_step(batch)
            # Log the loss so we can monitor test performance
            self.common_log_step({"loss": step_output["loss"]}, "test")
            return step_output

Using Your Custom Model
------------------------

Now that we've defined our model, let's see how to use it. This example shows how you would use the model in a Jupyter notebook:

Setting Up the Data
~~~~~~~~~~~~~~~~~~~~

First, we need to set up our data. The ``EncoderDecoderDataModule`` handles loading and preprocessing:

.. code-block:: python

    from transformertf.data import EncoderDecoderDataModule

    # Create the data module
    # This handles loading data from files and creating batches for training
    datamodule = EncoderDecoderDataModule(
        known_covariates="I_meas_A_filtered",           # The input feature (current)
        target_covariate="B_meas_T_filtered",           # What we want to predict (magnetic field)
        train_df_paths=["path/to/train_data.parquet"],  # Training data file
        val_df_paths=["path/to/val_data.parquet"],      # Validation data file
        ctxt_seq_len=1,                                 # Length of context sequence
        tgt_seq_len=1,                                  # Length of target sequence
        downsample=20,                                  # Downsample data by factor of 20
        batch_size=256,                                 # How many samples per batch
        num_workers=8,                                  # Number of workers for data loading
    )

    # Prepare the data (download, process, etc.)
    datamodule.prepare_data()
    # Set up the data (create train/val/test splits)
    datamodule.setup()

Creating and Training the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we create our model and train it:

.. code-block:: python

    # Create our custom model
    model = MLPHysteresis(
        num_layers=2,           # 2 hidden layers
        hidden_size=128,        # 128 neurons per layer
        dropout=0.1,            # 10% dropout
        lr=1e-4,                # Learning rate of 0.0001
        weight_decay=1e-4       # Small weight decay for regularization
    )

    # Set up the trainer
    # The trainer handles the training loop, validation, checkpointing, etc.
    import lightning as L

    trainer = L.Trainer(
        max_epochs=100,                     # Train for at most 100 epochs
        gradient_clip_val=1.0,              # Clip gradients to prevent exploding gradients
        logger=L.pytorch.loggers.TensorBoardLogger("logs", name="mlp_hysteresis"),
        callbacks=[
            # Stop training if validation loss doesn't improve for 10 epochs
            L.pytorch.callbacks.EarlyStopping(
                monitor="loss/validation",
                patience=10,
                mode="min"
            ),
            # Save the best model based on validation loss
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="loss/validation",
                save_top_k=1
            ),
        ]
    )

    # Train the model
    # This will run the training loop automatically
    trainer.fit(model, datamodule)

Loading the Best Model
~~~~~~~~~~~~~~~~~~~~~~

After training, we want to load the best model (the one with lowest validation loss):

.. code-block:: python

    # Get the path to the best checkpoint
    checkpoint_path = trainer.checkpoint_callback.best_model_path

    # Load the best model
    model = MLPHysteresis.load_from_checkpoint(checkpoint_path)

    # Run validation to get final metrics
    trainer.validate(model, dataloaders=[datamodule.val_dataloader()])

Making Predictions and Analyzing Results
----------------------------------------

After training, you can make predictions and analyze how well your model performs:

.. code-block:: python

    # Get the validation outputs (predictions and targets)
    # The model automatically collects these during validation
    validation_outputs = model.validation_outputs[0]

    # Extract all predictions and concatenate them
    predictions = torch.cat([output["point_prediction"] for output in validation_outputs], dim=0)

    # Extract all targets from the validation data
    targets = torch.cat([batch["target"] for batch in datamodule.val_dataloader()], dim=0)

    # Convert to numpy arrays for easier analysis
    predictions = predictions.squeeze().cpu().numpy()
    targets = targets.squeeze().cpu().numpy()

    # Create plots to visualize the results
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot predictions vs ground truth
    ax1.plot(targets, label="Ground truth", alpha=0.7)
    ax1.plot(predictions, label="Predictions", alpha=0.7)
    ax1.legend()
    ax1.set_ylabel("Magnetic Field")
    ax1.set_title("Model Predictions vs Ground Truth")

    # Plot residuals (errors)
    ax2.plot(targets - predictions, label="Residuals", alpha=0.7)
    ax2.legend()
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Sample")
    ax2.set_title("Prediction Errors")

    plt.tight_layout()
    plt.show()

    # Calculate some basic statistics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")

Understanding the Framework Components
--------------------------------------

Hyperparameter Management
~~~~~~~~~~~~~~~~~~~~~~~~~

Every model parameter you define in ``__init__`` is automatically saved when you call ``self.save_hyperparameters()``. This means:

- Lightning can restore your model exactly as it was
- You can access parameters later using ``self.hparams.parameter_name``
- The parameters are included in model checkpoints

Data Handling
~~~~~~~~~~~~~

The framework expects your model to work with specific data formats:

- **EncoderDecoderTargetSample**: Used for sequence-to-sequence tasks
- **TimeSeriesSample**: Used for basic time series prediction

These are just dictionaries with specific required fields like ``"encoder_input"``, ``"decoder_input"``, and ``"target"``.

Automatic Metrics
~~~~~~~~~~~~~~~~~

When you call ``self.common_log_step()``, the framework automatically calculates and logs several metrics:

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error

Model Compilation
~~~~~~~~~~~~~~~~

The ``compile_model`` parameter enables PyTorch 2.0 compilation, which can significantly speed up training on modern hardware. The base class handles this automatically.

Loss Functions
~~~~~~~~~~~~~

You initialize ``self.criterion`` in ``__init__`` with your chosen loss function. The framework supports:

- Standard PyTorch losses (MSELoss, L1Loss, etc.)
- Custom losses from the TransformerTF library (like QuantileLoss)

Advanced Features
-----------------

Using Quantile Loss for Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of just predicting a single value, you can predict multiple quantiles to get uncertainty estimates:

.. code-block:: python

    from transformertf.nn import QuantileLoss

    # In your model's __init__:
    self.criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])  # 10%, 50%, 90% quantiles

    # In your common_step method:
    def common_step(self, batch):
        # ... your existing code ...

        # For quantile loss, extract the point prediction (median)
        if hasattr(self.criterion, 'point_prediction'):
            point_pred = self.criterion.point_prediction(y_hat)
        else:
            point_pred = y_hat

        return {
            "loss": loss,
            "output": y_hat,
            "point_prediction": point_pred,
        }

Custom Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can implement custom learning rate schedules:

.. code-block:: python

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100  # Reduce learning rate over 100 epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/validation"  # Monitor validation loss
        }

Adding Custom Metrics
~~~~~~~~~~~~~~~~~~~~~

You can add your own metrics alongside the automatic ones:

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        step_output = self.common_step(batch)

        # Log the standard loss
        self.common_log_step({"loss": step_output["loss"]}, "validation")

        # Calculate and log a custom metric
        custom_metric = self.calculate_custom_metric(
            step_output["point_prediction"],
            batch["target"]
        )
        self.log("custom_metric/validation", custom_metric)

        return step_output

    def calculate_custom_metric(self, prediction, target):
        """Calculate a custom metric - normalized RMSE in this example."""
        rmse = torch.sqrt(torch.mean((prediction - target) ** 2))
        target_range = torch.max(target) - torch.min(target)
        return rmse / target_range

This comprehensive guide provides everything you need to extend the TransformerTF API with your own custom models, explained in detail for newcomers to the framework.
