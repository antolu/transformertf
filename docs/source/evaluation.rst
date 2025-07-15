Model Evaluation with Custom DataFrames
=======================================

This guide demonstrates how to evaluate trained models using custom DataFrames containing physics sensor data, based on the workflow shown in the LSTM preprocessing notebook.

Overview
--------

The evaluation workflow involves:

1. Loading a trained model and datamodule from a checkpoint
2. Creating a dataloader from your custom DataFrame containing sensor measurements
3. Running inference using the model's predict step
4. Post-processing predictions and visualizing results with physics units

Step-by-Step Guide
------------------

1. Import Required Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from __future__ import annotations

    import pathlib
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch

    from transformertf.models.lstm import LSTM
    from transformertf.data import TimeSeriesDataModule

2. Load Model and DataModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load your trained model and data module from a checkpoint:

.. code-block:: python

    # Path to your trained model checkpoint
    CKPT_PATH = pathlib.Path("~/path/to/your/model.ckpt").expanduser()

    # Load the datamodule and model
    datamodule = TimeSeriesDataModule.load_from_checkpoint(CKPT_PATH)
    model = LSTM.load_from_checkpoint(CKPT_PATH, map_location="cpu")

3. Prepare Your DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~

Load your custom DataFrame with sensor measurements for evaluation:

.. code-block:: python

    # Load your custom DataFrame with sensor data
    df = pd.read_parquet("path/to/your/sensor_data.parquet")
    # Or load from CSV: df = pd.read_csv("path/to/your/sensor_data.csv")

    # Create a dataloader for prediction
    dataloader = datamodule.make_dataloader(df, predict=True)

4. Run Model Inference
~~~~~~~~~~~~~~~~~~~~~~

Execute the prediction loop with proper model lifecycle management:

.. code-block:: python

    # Set model to evaluation mode
    model = model.eval()

    # Initialize prediction phase
    model.on_predict_start()
    model.on_predict_epoch_start()

    # Run inference on each batch
    for i, batch in enumerate(dataloader):
        model.on_predict_batch_start(batch, batch_idx=i)
        with torch.no_grad():
            outputs = model.predict_step(batch, batch_idx=i)
        model.on_predict_batch_end(batch, outputs, batch_idx=i)

    # Finalize prediction phase
    model.on_predict_epoch_end()
    model.on_predict_end()

5. Extract and Process Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collect prediction outputs and prepare for analysis:

.. code-block:: python

    # Get prediction outputs from the model
    prediction_outputs = model.inference_outputs[0]

    # Concatenate results from all batches
    inputs = torch.cat([output["input"].squeeze()[..., 0] for output in prediction_outputs])
    point_predictions = torch.cat([output["point_prediction"].squeeze() for output in prediction_outputs])
    targets = torch.cat([output["target"].squeeze() for output in prediction_outputs])

    # Get original data length for trimming
    total_points = len(df)

6. Apply Inverse Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert normalized predictions back to original scale:

.. code-block:: python

    # Apply inverse transforms to get original scale
    inputs = datamodule.transforms[datamodule.known_covariates[0].name].inverse_transform(inputs)
    predictions = datamodule.target_transform.inverse_transform(inputs, point_predictions)
    targets = datamodule.target_transform.inverse_transform(inputs, targets)

    # Convert to numpy and trim to original length
    inputs = inputs.cpu().numpy()[:total_points]
    predictions = predictions.cpu().numpy()[:total_points]
    targets = targets.cpu().numpy()[:total_points]

7. Visualize Results
~~~~~~~~~~~~~~~~~~~~

Create plots to analyze model performance with physics units:

.. code-block:: python

    # Plot predictions vs targets for physics measurements
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(inputs, targets, label="Measured Values", alpha=0.7)
    ax.plot(inputs, predictions, label="Model Predictions", alpha=0.7)
    ax.set_xlabel("Voltage Input (V)")
    ax.set_ylabel("Magnetic Field (mT)")
    ax.legend()
    ax.set_title("Physics Model: Predicted vs Measured Magnetic Field")
    ax.grid(True, alpha=0.3)
    plt.show()

    # Plot residuals for error analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(inputs, targets - predictions, label="Prediction Residuals", color="red")
    ax.set_xlabel("Voltage Input (V)")
    ax.set_ylabel("Residual Error (mT)")
    ax.legend()
    ax.set_title("Model Prediction Errors")
    ax.grid(True, alpha=0.3)
    plt.show()

Complete Example
----------------

Here's a complete example combining all steps:

.. code-block:: python

    from __future__ import annotations

    import pathlib
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch

    from transformertf.models.lstm import LSTM
    from transformertf.data import TimeSeriesDataModule

    # Path to your trained model checkpoint
    CKPT_PATH = pathlib.Path("~/path/to/your/model.ckpt").expanduser()

    # Load model and datamodule
    datamodule = TimeSeriesDataModule.load_from_checkpoint(CKPT_PATH)
    model = LSTM.load_from_checkpoint(CKPT_PATH, map_location="cpu")

    # Load custom DataFrame with sensor data
    df = pd.read_parquet("path/to/your/sensor_data.parquet")
    dataloader = datamodule.make_dataloader(df, predict=True)

    # Run inference
    model = model.eval()
    model.on_predict_start()
    model.on_predict_epoch_start()

    for i, batch in enumerate(dataloader):
        model.on_predict_batch_start(batch, batch_idx=i)
        with torch.no_grad():
            outputs = model.predict_step(batch, batch_idx=i)
        model.on_predict_batch_end(batch, outputs, batch_idx=i)

    model.on_predict_epoch_end()
    model.on_predict_end()

    # Process results
    prediction_outputs = model.inference_outputs[0]
    inputs = torch.cat([output["input"].squeeze()[..., 0] for output in prediction_outputs])
    point_predictions = torch.cat([output["point_prediction"].squeeze() for output in prediction_outputs])
    targets = torch.cat([output["target"].squeeze() for output in prediction_outputs])

    # Apply inverse transforms
    inputs = datamodule.transforms[datamodule.known_covariates[0].name].inverse_transform(inputs)
    predictions = datamodule.target_transform.inverse_transform(inputs, point_predictions)
    targets = datamodule.target_transform.inverse_transform(inputs, targets)

    # Convert to numpy
    total_points = len(df)
    inputs = inputs.cpu().numpy()[:total_points]
    predictions = predictions.cpu().numpy()[:total_points]
    targets = targets.cpu().numpy()[:total_points]

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Predictions vs targets
    ax1.plot(inputs, targets, label="Ground Truth", alpha=0.7)
    ax1.plot(inputs, predictions, label="Predictions", alpha=0.7)
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")
    ax1.legend()
    ax1.set_title("Model Predictions vs Ground Truth")
    ax1.grid(True, alpha=0.3)

    # Residuals
    ax2.plot(inputs, targets - predictions, label="Residuals", color="red")
    ax2.set_xlabel("Input")
    ax2.set_ylabel("Residual")
    ax2.legend()
    ax2.set_title("Prediction Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Important Notes
---------------

**Model Lifecycle Management**
    Always call the prediction lifecycle methods (``on_predict_start``, ``on_predict_epoch_start``, etc.) to ensure proper model state management, especially for models with hidden states like LSTM and GRU.

**Data Compatibility**
    Ensure your custom DataFrame has the same column structure and data preprocessing as the training data. The model expects data in the same format it was trained on.

**Transform Consistency**
    The datamodule's transforms must match those used during training. Loading from checkpoint ensures this consistency.

**Memory Management**
    Use ``torch.no_grad()`` during inference to prevent gradient computation and reduce memory usage.

**Batch Processing**
    The model processes data in batches. Results are concatenated from all batches, so ensure you trim to the original data length.

**GPU/CPU Considerations**
    Specify ``map_location="cpu"`` when loading checkpoints if you want to run inference on CPU, or adjust accordingly for GPU inference.

Troubleshooting
---------------

**Common Issues:**

1. **Shape Mismatches**: Ensure your DataFrame columns match the expected input features
2. **Transform Errors**: Verify that the data preprocessing matches training conditions
3. **Memory Issues**: Consider reducing batch size or using CPU for large datasets
4. **Hidden State Issues**: Ensure proper model lifecycle management for recurrent models

**Performance Optimization:**

- Use GPU if available for faster inference
- Batch your data appropriately for your hardware
- Consider using model compilation for PyTorch 2.0+ for improved performance

This workflow provides a robust foundation for evaluating trained models on custom datasets while maintaining consistency with the training pipeline.
