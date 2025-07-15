Basic Physics Signal Forecasting
=================================

This tutorial demonstrates end-to-end physics signal forecasting using the Temporal Fusion Transformer (TFT). We'll generate synthetic sinusoidal sensor data, train a model, and make predictions with uncertainty quantification.

Overview
--------

**What you'll learn:**
- How to structure high-frequency sensor data for TransformerTF
- How to configure and train a TFT model for physics applications
- How to generate predictions with confidence intervals
- How to visualize and interpret results

**Prerequisites:**
- TransformerTF installed (``pip install transformertf``)
- Basic familiarity with signal processing concepts
- Python data science environment (pandas, numpy, matplotlib)

Step 1: Generate Synthetic Data
-------------------------------

First, let's create realistic synthetic sensor data with sinusoidal patterns and physics-based relationships:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from datetime import datetime, timedelta

   # Set random seed for reproducibility
   np.random.seed(42)

   # Generate high-frequency sensor data (10 Hz for 1 hour)
   start_date = datetime(2022, 1, 1)
   end_date = datetime(2022, 1, 1, 1)  # 1 hour of data
   timestamps = pd.date_range(start_date, end_date, freq='100ms')  # 10 Hz sampling
   n_points = len(timestamps)

   # Time in seconds for signal generation
   time_seconds = np.arange(n_points) / 10.0  # 10 Hz = 0.1 second intervals

   # Generate synthetic voltage input (sinusoidal with multiple frequency components)
   # 1. Primary sinusoidal component
   voltage_primary = 5 * np.sin(2 * np.pi * 0.1 * time_seconds)  # 0.1 Hz sine wave

   # 2. Secondary harmonic
   voltage_secondary = 1.5 * np.sin(2 * np.pi * 0.3 * time_seconds)  # 0.3 Hz component

   # 3. High-frequency noise
   voltage_noise = 0.2 * np.random.normal(0, 1, n_points)

   # Combine voltage components
   voltage_input = voltage_primary + voltage_secondary + voltage_noise

   # Generate temperature variations (slow thermal response)
   temperature = 20 + 5 * np.sin(2 * np.pi * 0.01 * time_seconds) + np.random.normal(0, 0.5, n_points)

   # Generate magnetic field response (nonlinear relationship with voltage)
   magnetic_field_linear = 0.8 * voltage_input  # Linear component
   magnetic_field_nonlinear = 0.2 * np.sin(voltage_input)  # Nonlinear component
   temperature_coupling = 0.1 * (temperature - 20) * voltage_input  # Temperature coupling
   measurement_noise = np.random.normal(0, 0.1, n_points)

   # Combine all components for target
   magnetic_field = magnetic_field_linear + magnetic_field_nonlinear + temperature_coupling + measurement_noise

   # Create feature columns
   df = pd.DataFrame({
       'timestamp': timestamps,
       'magnetic_field': magnetic_field,
       'voltage_input': voltage_input,
       'temperature': temperature,
       'frequency': 0.1,  # Primary frequency component
       'amplitude': 5.0,  # Primary amplitude
       'sensor_id': 'sensor_1'  # Required for multi-entity support
   })

   print(f"Generated {len(df)} data points")
   print(f"Date range: {df.timestamp.min()} to {df.timestamp.max()}")
   print(f"Magnetic field statistics: mean={df.magnetic_field.mean():.2f}, std={df.magnetic_field.std():.2f}")
   print(f"Voltage input statistics: mean={df.voltage_input.mean():.2f}, std={df.voltage_input.std():.2f}")

**Visualize the generated data:**

.. code-block:: python

   # Plot first 100 seconds of data to see patterns
   sample_data = df.iloc[:1000]  # First 1000 points (100 seconds at 10 Hz)

   fig, axes = plt.subplots(3, 1, figsize=(12, 10))

   # Magnetic field (target)
   axes[0].plot(sample_data.timestamp, sample_data.magnetic_field, 'b-', linewidth=1)
   axes[0].set_title('Magnetic Field Response (Target)')
   axes[0].set_ylabel('Magnetic Field (mT)')
   axes[0].grid(True, alpha=0.3)

   # Voltage input
   axes[1].plot(sample_data.timestamp, sample_data.voltage_input, 'r-', linewidth=1)
   axes[1].set_title('Voltage Input Signal')
   axes[1].set_ylabel('Voltage (V)')
   axes[1].grid(True, alpha=0.3)

   # Temperature
   axes[2].plot(sample_data.timestamp, sample_data.temperature, 'g-', linewidth=1)
   axes[2].set_title('Temperature Variation')
   axes[2].set_ylabel('Temperature (°C)')
   axes[2].set_xlabel('Time')
   axes[2].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Step 2: Prepare Data for Training
---------------------------------

Split the data chronologically and save in Parquet format:

.. code-block:: python

   # Calculate split points (80% train, 10% validation, 10% test)
   n_total = len(df)
   train_end = int(0.8 * n_total)
   val_end = int(0.9 * n_total)

   # Split data chronologically (important for sensor data!)
   train_df = df.iloc[:train_end].copy()
   val_df = df.iloc[train_end:val_end].copy()
   test_df = df.iloc[val_end:].copy()

   print(f"Train: {len(train_df)} samples ({train_df.timestamp.min()} to {train_df.timestamp.max()})")
   print(f"Validation: {len(val_df)} samples ({val_df.timestamp.min()} to {val_df.timestamp.max()})")
   print(f"Test: {len(test_df)} samples ({test_df.timestamp.min()} to {test_df.timestamp.max()})")

   # Save to Parquet files
   train_df.to_parquet('sensor_train.parquet', index=False)
   val_df.to_parquet('sensor_val.parquet', index=False)
   test_df.to_parquet('sensor_test.parquet', index=False)

   print("Sensor data saved to Parquet files")

Step 3: Create Configuration File
---------------------------------

Create a YAML configuration for the TFT model with quantile regression:

.. code-block:: python

   config_yaml = """
   # TFT Configuration for Physics Signal Forecasting
   seed_everything: 42

   trainer:
     max_epochs: 50
     accelerator: auto
     devices: auto
     gradient_clip_val: 1.0
     check_val_every_n_epoch: 1

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       n_dim_model: 64
       hidden_continuous_dim: 16
       num_heads: 4
       num_lstm_layers: 2
       dropout: 0.1
       quantiles: [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
       criterion:
         class_path: transformertf.nn.QuantileLoss

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ["sensor_train.parquet"]
       val_df_paths: ["sensor_val.parquet"]
       target_covariate: "magnetic_field"
       known_covariates: ["voltage_input", "temperature", "frequency", "amplitude"]
       static_categorical_variables: ["sensor_id"]
       ctxt_seq_len: 500   # 50 seconds of context at 10 Hz
       tgt_seq_len: 100    # Predict 10 seconds ahead
       batch_size: 32
       normalize: true
       num_workers: 0      # Use 0 for tutorial compatibility

   optimizer:
     class_path: torch.optim.Adam
     init_args:
       lr: 0.001
       weight_decay: 1e-4

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
     init_args:
       mode: min
       factor: 0.5
       patience: 5
       verbose: true
   """

   # Save configuration
   with open('tft_config.yml', 'w') as f:
       f.write(config_yaml)

   print("Configuration saved to tft_config.yml")

Step 4: Train the Model
-----------------------

Train the TFT model using the Lightning CLI:

.. code-block:: bash

   # Train the model
   transformertf fit --config tft_config.yml

**Alternative: Using Python API**

.. code-block:: python

   import lightning as L
   from transformertf.data import EncoderDecoderDataModule
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
   from transformertf.nn import QuantileLoss

   # Initialize data module
   data_module = EncoderDecoderDataModule(
       train_df_paths=["sensor_train.parquet"],
       val_df_paths=["sensor_val.parquet"],
       target_covariate="magnetic_field",
       known_covariates=["voltage_input", "temperature", "frequency", "amplitude"],
       static_categorical_variables=["sensor_id"],
       ctxt_seq_len=500,  # 50 seconds at 10 Hz
       tgt_seq_len=100,   # 10 seconds prediction
       batch_size=32,
       normalize=True
   )

   # Initialize model
   model = TemporalFusionTransformer(
       n_dim_model=64,
       hidden_continuous_dim=16,
       num_heads=4,
       num_lstm_layers=2,
       dropout=0.1,
       quantiles=[0.1, 0.5, 0.9],
       criterion=QuantileLoss()
   )

   # Setup trainer with callbacks
   trainer = L.Trainer(
       max_epochs=50,
       accelerator="auto",
       gradient_clip_val=1.0,
       callbacks=[
           L.callbacks.ModelCheckpoint(
               monitor="validation/loss",
               mode="min",
               save_top_k=1,
               filename="best-{epoch}-{validation/loss:.4f}"
           ),
           L.callbacks.EarlyStopping(
               monitor="validation/loss",
               patience=10,
               mode="min"
           ),
           L.callbacks.LearningRateMonitor(logging_interval="epoch")
       ]
   )

   # Train the model
   trainer.fit(model, data_module)

   print(f"Training completed. Best model saved at: {trainer.checkpoint_callback.best_model_path}")

Step 5: Generate Predictions
----------------------------

Use the trained model to generate predictions on test data:

.. code-block:: python

   from transformertf.utils.predict import predict

   # Load best checkpoint
   best_model_path = trainer.checkpoint_callback.best_model_path

   # Create test data module (exclude target from known covariates for prediction)
   test_data_module = EncoderDecoderDataModule(
       train_df_paths=["sensor_test.parquet"],  # Use test data as "train" for prediction
       target_covariate="magnetic_field",
       known_covariates=["voltage_input", "temperature", "frequency", "amplitude"],
       static_categorical_variables=["sensor_id"],
       ctxt_seq_len=500,  # 50 seconds at 10 Hz
       tgt_seq_len=100,   # 10 seconds prediction
       batch_size=32,
       normalize=True
   )

   # Generate predictions
   predictions = predict(
       model_ckpt_path=best_model_path,
       datamodule=test_data_module,
       trainer=trainer
   )

   print(f"Generated {len(predictions)} prediction batches")

**Alternative: Direct model prediction**

.. code-block:: python

   # Load the best model
   model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
   model.eval()

   # Get predictions on test data
   test_dataloader = test_data_module.test_dataloader()
   predictions = trainer.predict(model, test_dataloader)

   # Convert predictions to numpy arrays
   pred_quantiles = []
   actuals = []

   for batch_pred, batch in zip(predictions, test_dataloader):
       pred_quantiles.append(batch_pred.cpu().numpy())
       actuals.append(batch['decoder_target'].cpu().numpy())

   pred_quantiles = np.concatenate(pred_quantiles, axis=0)
   actuals = np.concatenate(actuals, axis=0)

   print(f"Prediction shape: {pred_quantiles.shape}")  # [n_samples, seq_len, n_quantiles]
   print(f"Actual shape: {actuals.shape}")  # [n_samples, seq_len]

Step 6: Visualize Results
------------------------

Create visualizations to evaluate model performance:

.. code-block:: python

   # Select a few samples for visualization
   n_samples_to_plot = 5
   sample_indices = np.random.choice(len(pred_quantiles), n_samples_to_plot, replace=False)

   fig, axes = plt.subplots(n_samples_to_plot, 1, figsize=(12, 3 * n_samples_to_plot))
   if n_samples_to_plot == 1:
       axes = [axes]

   for i, idx in enumerate(sample_indices):
       ax = axes[i]

       # Get predictions and actuals for this sample
       pred_lower = pred_quantiles[idx, :, 0]  # 10th percentile
       pred_median = pred_quantiles[idx, :, 1]  # 50th percentile (median)
       pred_upper = pred_quantiles[idx, :, 2]   # 90th percentile
       actual = actuals[idx, :]

       # Time axis for plotting
       time_steps = range(len(actual))

       # Plot actual values
       ax.plot(time_steps, actual, 'b-', label='Actual', linewidth=2)

       # Plot median prediction
       ax.plot(time_steps, pred_median, 'r--', label='Prediction (median)', linewidth=2)

       # Plot confidence interval
       ax.fill_between(time_steps, pred_lower, pred_upper,
                      alpha=0.3, color='red', label='80% Confidence Interval')

       ax.set_title(f'Sample {idx + 1}: 10-second Magnetic Field Forecast')
       ax.set_xlabel('Time Steps (0.1s intervals)')
       ax.set_ylabel('Magnetic Field (mT)')
       ax.legend()
       ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

**Calculate evaluation metrics:**

.. code-block:: python

   from sklearn.metrics import mean_absolute_error, mean_squared_error

   # Flatten for metric calculation
   pred_median_flat = pred_quantiles[:, :, 1].flatten()  # Use median prediction
   actual_flat = actuals.flatten()

   # Calculate metrics
   mae = mean_absolute_error(actual_flat, pred_median_flat)
   rmse = np.sqrt(mean_squared_error(actual_flat, pred_median_flat))
   mape = np.mean(np.abs((actual_flat - pred_median_flat) / actual_flat)) * 100

   print(f"Evaluation Metrics:")
   print(f"MAE: {mae:.3f}")
   print(f"RMSE: {rmse:.3f}")
   print(f"MAPE: {mape:.2f}%")

   # Calculate coverage of confidence intervals
   in_interval = (actual_flat >= pred_quantiles[:, :, 0].flatten()) & \
                 (actual_flat <= pred_quantiles[:, :, 2].flatten())
   coverage = np.mean(in_interval) * 100

   print(f"80% Confidence Interval Coverage: {coverage:.1f}%")

Step 7: Model Interpretation
----------------------------

TFT provides interpretable outputs including attention weights and variable importance:

.. code-block:: python

   # Get a single batch for interpretation
   model.eval()
   sample_batch = next(iter(test_dataloader))

   with torch.no_grad():
       # Forward pass with interpretation
       output = model(sample_batch)

       # Get attention weights (if available)
       if hasattr(model, 'get_attention_weights'):
           attention_weights = model.get_attention_weights(sample_batch)

           # Plot attention heatmap for first sample
           plt.figure(figsize=(10, 6))
           plt.imshow(attention_weights[0].cpu().numpy(), aspect='auto', cmap='Blues')
           plt.title('Temporal Attention Weights')
           plt.xlabel('Time Steps')
           plt.ylabel('Attention Heads')
           plt.colorbar()
           plt.show()

Next Steps
----------

**Congratulations!** You've successfully:

1. ✅ Generated and prepared time series data
2. ✅ Configured and trained a TFT model
3. ✅ Generated predictions with uncertainty quantification
4. ✅ Visualized and evaluated results

**What to explore next:**

- **Hyperparameter Tuning**: Use :doc:`../usage` guide for Ray Tune optimization
- **Multiple Time Series**: Extend to multiple entities/locations
- **Advanced Features**: Try static categorical variables and custom transforms
- **Production Deployment**: See :doc:`05_production_deployment` for serving models

**Common Issues:**

- **Memory errors**: Reduce ``batch_size`` or ``ctxt_seq_len``
- **Poor convergence**: Try different learning rates or longer training
- **NaN losses**: Check data for missing values or infinities

For more advanced usage patterns, see the :doc:`../examples` gallery and :doc:`../usage` guide.
