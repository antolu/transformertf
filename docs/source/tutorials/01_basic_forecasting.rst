Basic Time Series Forecasting
==============================

This tutorial demonstrates end-to-end time series forecasting using the Temporal Fusion Transformer (TFT). We'll generate synthetic data, train a model, and make predictions with uncertainty quantification.

Overview
--------

**What you'll learn:**
- How to structure time series data for TransformerTF
- How to configure and train a TFT model
- How to generate predictions with confidence intervals
- How to visualize and interpret results

**Prerequisites:**
- TransformerTF installed (``pip install transformertf``)
- Basic familiarity with time series concepts
- Python data science environment (pandas, numpy, matplotlib)

Step 1: Generate Synthetic Data
-------------------------------

First, let's create realistic synthetic time series data with multiple features and seasonal patterns:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from datetime import datetime, timedelta

   # Set random seed for reproducibility
   np.random.seed(42)

   # Generate 2 years of hourly data
   start_date = datetime(2022, 1, 1)
   end_date = datetime(2023, 12, 31, 23)
   timestamps = pd.date_range(start_date, end_date, freq='H')
   n_points = len(timestamps)

   # Create time-based features
   hours = timestamps.hour
   days = timestamps.dayofyear
   months = timestamps.month

   # Generate synthetic target with multiple patterns
   # 1. Daily cycle
   daily_pattern = 10 * np.sin(2 * np.pi * hours / 24)

   # 2. Weekly cycle
   weekly_pattern = 5 * np.sin(2 * np.pi * days / 7)

   # 3. Annual trend
   annual_trend = 0.01 * days

   # 4. Random noise
   noise = np.random.normal(0, 2, n_points)

   # 5. External feature influence
   temperature = 20 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 3, n_points)
   demand_influence = 0.3 * temperature

   # Combine all components
   target = 100 + daily_pattern + weekly_pattern + annual_trend + demand_influence + noise

   # Create feature columns
   df = pd.DataFrame({
       'timestamp': timestamps,
       'target': target,
       'temperature': temperature,
       'hour_of_day': hours,
       'day_of_week': timestamps.dayofweek,
       'month': months,
       'is_weekend': (timestamps.dayofweek >= 5).astype(int),
       'entity_id': 'location_1'  # Required for multi-entity support
   })

   print(f"Generated {len(df)} data points")
   print(f"Date range: {df.timestamp.min()} to {df.timestamp.max()}")
   print(f"Target statistics: mean={df.target.mean():.2f}, std={df.target.std():.2f}")

**Visualize the generated data:**

.. code-block:: python

   # Plot first week of data to see patterns
   week_data = df.iloc[:168]  # First 168 hours (1 week)

   fig, axes = plt.subplots(2, 1, figsize=(12, 8))

   # Target variable
   axes[0].plot(week_data.timestamp, week_data.target)
   axes[0].set_title('Target Variable (First Week)')
   axes[0].set_ylabel('Target Value')

   # Temperature feature
   axes[1].plot(week_data.timestamp, week_data.temperature, color='orange')
   axes[1].set_title('Temperature Feature (First Week)')
   axes[1].set_ylabel('Temperature')
   axes[1].set_xlabel('Time')

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

   # Split data chronologically (important for time series!)
   train_df = df.iloc[:train_end].copy()
   val_df = df.iloc[train_end:val_end].copy()
   test_df = df.iloc[val_end:].copy()

   print(f"Train: {len(train_df)} samples ({train_df.timestamp.min()} to {train_df.timestamp.max()})")
   print(f"Validation: {len(val_df)} samples ({val_df.timestamp.min()} to {val_df.timestamp.max()})")
   print(f"Test: {len(test_df)} samples ({test_df.timestamp.min()} to {test_df.timestamp.max()})")

   # Save to Parquet files
   train_df.to_parquet('train_data.parquet', index=False)
   val_df.to_parquet('val_data.parquet', index=False)
   test_df.to_parquet('test_data.parquet', index=False)

   print("Data saved to Parquet files")

Step 3: Create Configuration File
---------------------------------

Create a YAML configuration for the TFT model with quantile regression:

.. code-block:: python

   config_yaml = """
   # TFT Configuration for Basic Forecasting
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
       train_df_paths: ["train_data.parquet"]
       val_df_paths: ["val_data.parquet"]
       target_covariate: "target"
       known_covariates: ["temperature", "hour_of_day", "day_of_week", "month", "is_weekend"]
       static_categorical_variables: ["entity_id"]
       ctxt_seq_len: 168   # 1 week of context (168 hours)
       tgt_seq_len: 24     # Predict 24 hours ahead
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
       train_df_paths=["train_data.parquet"],
       val_df_paths=["val_data.parquet"],
       target_covariate="target",
       known_covariates=["temperature", "hour_of_day", "day_of_week", "month", "is_weekend"],
       static_categorical_variables=["entity_id"],
       ctxt_seq_len=168,
       tgt_seq_len=24,
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
       train_df_paths=["test_data.parquet"],  # Use test data as "train" for prediction
       target_covariate="target",
       known_covariates=["temperature", "hour_of_day", "day_of_week", "month", "is_weekend"],
       static_categorical_variables=["entity_id"],
       ctxt_seq_len=168,
       tgt_seq_len=24,
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

       ax.set_title(f'Sample {idx + 1}: 24-hour Forecast')
       ax.set_xlabel('Hours')
       ax.set_ylabel('Target Value')
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
