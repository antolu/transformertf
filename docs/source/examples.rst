Examples Gallery
================

This gallery showcases practical examples for common time series forecasting scenarios with TransformerTF. Each example includes complete code and configuration files.

Quick Examples
--------------

Simple Sinusoidal Signal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict sinusoidal magnetic field response using LSTM:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from transformertf.data import TimeSeriesDataModule
   from transformertf.models.lstm import LSTM
   import lightning as L

   # Generate sinusoidal sensor data
   timestamps = pd.date_range('2022-01-01', periods=10000, freq='100ms')  # 10 Hz sampling
   time_vals = np.arange(len(timestamps)) / 10.0  # Convert to seconds
   voltage_input = 5 * np.sin(2 * np.pi * 0.1 * time_vals)  # 0.1 Hz sine wave
   magnetic_field = 0.8 * voltage_input + 0.2 * np.sin(voltage_input) + np.random.normal(0, 0.1, len(timestamps))
   df = pd.DataFrame({'timestamp': timestamps, 'voltage_input': voltage_input, 'magnetic_field': magnetic_field})

   # Save train/val splits
   train_df = df.iloc[:8000]
   val_df = df.iloc[8000:]
   train_df.to_parquet('train.parquet', index=False)
   val_df.to_parquet('val.parquet', index=False)

   # Configure data and model
   data_module = TimeSeriesDataModule(
       train_df_paths=['train.parquet'],
       val_df_paths=['val.parquet'],
       target_covariate='magnetic_field',
       seq_len=100,  # 10 seconds at 10 Hz
       batch_size=16
   )

   model = LSTM(d_model=64, num_layers=2)

   # Train
   trainer = L.Trainer(max_epochs=50, accelerator='auto')
   trainer.fit(model, data_module)

**Configuration file equivalent:**

.. code-block:: yaml

   # simple_lstm.yml
   trainer:
     max_epochs: 50
     accelerator: auto

   model:
     class_path: transformertf.models.lstm.LSTM
     init_args:
       hidden_size: 64
       num_layers: 2

   data:
     class_path: transformertf.data.TimeSeriesDataModule
     init_args:
       train_df_paths: ['train.parquet']
       val_df_paths: ['val.parquet']
       target_covariate: 'magnetic_field'
       seq_len: 100  # 10 seconds at 10 Hz
       batch_size: 16

**Usage:** ``transformertf fit --config simple_lstm.yml``

Multi-step Forecasting with TFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict multiple steps ahead with uncertainty:

.. code-block:: yaml

   # tft_multistep.yml
   seed_everything: 42

   trainer:
     max_epochs: 100
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       d_model: 64
       num_heads: 4
       quantiles: [0.1, 0.5, 0.9]
       criterion:
         class_path: transformertf.nn.QuantileLoss

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['sensor_train.parquet']
       val_df_paths: ['sensor_val.parquet']
       target_covariate: 'magnetic_field'
       known_covariates: ['voltage_input', 'temperature', 'frequency']
       ctxt_seq_len: 1000   # 100 seconds context at 10 Hz
       tgt_seq_len: 100     # 10 seconds prediction
       batch_size: 32

Real-World Scenarios
--------------------

Sensor Calibration and Drift Compensation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-sensor calibration with temperature compensation:

.. code-block:: python

   # Data preparation for sensor calibration
   import pandas as pd
   import numpy as np

   # Generate multi-sensor calibration data
   np.random.seed(42)
   sensors = ['sensor_A', 'sensor_B', 'sensor_C']
   timestamps = pd.date_range('2020-01-01', '2023-12-31', freq='100ms')

   data = []
   for sensor in sensors:
       # Sensor-specific characteristics
       sensitivity = {'sensor_A': 1.0, 'sensor_B': 0.95, 'sensor_C': 1.05}[sensor]
       offset = {'sensor_A': 0.0, 'sensor_B': 0.02, 'sensor_C': -0.01}[sensor]

       for i, timestamp in enumerate(timestamps):
           # True physical signal (sinusoidal input)
           time_sec = i * 0.1  # 10 Hz sampling
           true_signal = 5 * np.sin(2 * np.pi * 0.01 * time_sec)  # 0.01 Hz sine wave

           # Temperature variations
           temp_variation = 20 + 10 * np.sin(2 * np.pi * time_sec / 86400)  # Daily temperature cycle

           # Sensor response with temperature drift
           temp_coefficient = 0.001  # 0.1% per degree
           temp_drift = temp_coefficient * (temp_variation - 20)

           # Measured signal with sensor characteristics
           measured_signal = sensitivity * (1 + temp_drift) * true_signal + offset

           # Add measurement noise
           noise = np.random.normal(0, 0.05)
           measured_signal += noise

           data.append({
               'timestamp': timestamp,
               'sensor_id': sensor,
               'measured_signal': measured_signal,
               'true_signal': true_signal,
               'temperature': temp_variation,
               'voltage_input': true_signal * 0.2,  # Input voltage proportional to signal
               'frequency': 0.01 if i % 1000 < 500 else 0.02  # Step change in frequency
           })

   df = pd.DataFrame(data)

   # Split data chronologically
   train_end = '2022-12-31'
   val_end = '2023-06-30'

   train_df = df[df.timestamp <= train_end]
   val_df = df[(df.timestamp > train_end) & (df.timestamp <= val_end)]
   test_df = df[df.timestamp > val_end]

   # Save datasets
   train_df.to_parquet('sensor_train.parquet', index=False)
   val_df.to_parquet('sensor_val.parquet', index=False)
   test_df.to_parquet('sensor_test.parquet', index=False)

**Configuration:**

.. code-block:: yaml

   # sensor_calibration.yml
   seed_everything: 42

   trainer:
     max_epochs: 150
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       d_model: 128
       hidden_continuous_dim: 32
       num_heads: 8
       num_lstm_layers: 2
       dropout: 0.1
       quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
       criterion:
         class_path: transformertf.nn.QuantileLoss

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['sensor_train.parquet']
       val_df_paths: ['sensor_val.parquet']
       target_covariate: 'true_signal'
       known_covariates:
         - 'measured_signal'
         - 'temperature'
         - 'voltage_input'
         - 'frequency'
       static_categorical_variables: ['sensor_id']
       ctxt_seq_len: 1000   # 100 seconds context at 10 Hz
       tgt_seq_len: 100     # 10 seconds prediction
       batch_size: 32
       normalize: true

   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001
       weight_decay: 0.01

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
     init_args:
       mode: min
       factor: 0.5
       patience: 10

Temperature Sensor Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-frequency temperature measurement with environmental factors:

.. code-block:: yaml

   # temperature_sensor.yml
   seed_everything: 42

   trainer:
     max_epochs: 100
     accelerator: auto
     gradient_clip_val: 1.0
     precision: 16        # Mixed precision for efficiency

   model:
     class_path: transformertf.models.tsmixer.TSMixer
     init_args:
       n_block: 8
       ff_dim: 256
       dropout: 0.1
       activation: 'gelu'

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['temperature_train.parquet']
       val_df_paths: ['temperature_val.parquet']
       target_covariate: 'temperature'
       known_covariates:
         - 'voltage_input'
         - 'ambient_temperature'
         - 'humidity'
         - 'pressure'
         - 'heating_power'
       ctxt_seq_len: 1000   # 100 seconds at 10 Hz
       tgt_seq_len: 100     # 10 seconds prediction
       batch_size: 64
       normalize: true
       num_workers: 4

Vibration Analysis
~~~~~~~~~~~~~~~~~~

Accelerometer data processing with frequency analysis:

.. code-block:: yaml

   # vibration_analysis.yml
   seed_everything: 42

   trainer:
     max_epochs: 200
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.transformer.VanillaTransformer
     init_args:
       d_model: 256
       num_heads: 8
       num_encoder_layers: 6
       num_decoder_layers: 6
       ff_dim: 1024
       dropout: 0.15

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['vibration_train.parquet']
       val_df_paths: ['vibration_val.parquet']
       target_covariate: 'displacement'
       known_covariates:
         - 'acceleration_x'
         - 'acceleration_y'
         - 'acceleration_z'
         - 'frequency'
         - 'amplitude'
       static_categorical_variables: ['sensor_location', 'measurement_axis']
       ctxt_seq_len: 1000   # 100 seconds at 10 Hz
       tgt_seq_len: 100     # 10 seconds prediction
       batch_size: 16
       normalize: true

Specialized Applications
-----------------------

Physics-Informed Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~

Magnetic field modeling with physics constraints:

.. code-block:: yaml

   # physics_magnetic_field.yml
   seed_everything: 42

   trainer:
     max_epochs: 100
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.pete.PETE
     init_args:
       d_model: 128
       num_heads: 8
       num_layers: 3
       dropout: 0.1

   data:
     class_path: transformertf.data.TimeSeriesDataModule
     init_args:
       train_df_paths: ['magnetic_field_train.parquet']
       val_df_paths: ['magnetic_field_val.parquet']
       target_covariate: 'magnetic_field'
       known_covariates: ['voltage_input', 'current', 'temperature']
       seq_len: 500  # 50 seconds at 10 Hz
       batch_size: 32
       extra_transforms:
         magnetic_field:
           - class_path: transformertf.data.transform.DiscreteFunctionTransform
             init_args:
               x: 'calibration_function.csv'

Transfer Learning Example
~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune a pre-trained model on new domain:

.. code-block:: yaml

   # transfer_learning.yml
   transfer_ckpt: 'pretrained_models/retail_model.ckpt'

   seed_everything: 42

   trainer:
     max_epochs: 50       # Fewer epochs for fine-tuning
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       # Architecture must match pre-trained model
       d_model: 128
       num_heads: 8
       num_lstm_layers: 2
       dropout: 0.1

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['new_domain_train.parquet']
       val_df_paths: ['new_domain_val.parquet']
       # Data schema must be compatible
       target_covariate: 'target'
       known_covariates: ['feature1', 'feature2']
       ctxt_seq_len: 90
       tgt_seq_len: 30
       batch_size: 32

   optimizer:
     class_path: torch.optim.Adam
     init_args:
       lr: 0.0001         # Lower learning rate for fine-tuning

Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Automated hyperparameter search with Ray Tune:

.. code-block:: python

   from transformertf.utils.tune import tune, TuneConfig

   # Define search space
   search_space = {
       "model.init_args.d_model": {
           "type": "choice",
           "values": [32, 64, 128]
       },
       "model.init_args.num_heads": {
           "type": "choice",
           "values": [4, 8]
       },
       "model.init_args.dropout": {
           "type": "uniform",
           "low": 0.1,
           "high": 0.3
       },
       "optimizer.init_args.lr": {
           "type": "loguniform",
           "low": 1e-4,
           "high": 1e-2
       },
       "data.init_args.batch_size": {
           "type": "choice",
           "values": [16, 32, 64]
       }
   }

   # Configure tuning
   tune_config = TuneConfig(
       base_config="base_config.yml",
       search_space=search_space,
       num_samples=30,
       max_epochs=50,
       metric="validation/loss",
       mode="min"
   )

   # Run optimization
   best_config = tune(tune_config)
   print(f"Best configuration: {best_config}")

**Base configuration file:**

.. code-block:: yaml

   # base_config.yml
   seed_everything: 42

   trainer:
     max_epochs: 50
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     # Hyperparameters will be filled by tuning

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['data/train.parquet']
       val_df_paths: ['data/val.parquet']
       target_covariate: 'target'
       ctxt_seq_len: 100
       tgt_seq_len: 20
       # batch_size will be tuned

Production Deployment
--------------------

Model Serving Setup
~~~~~~~~~~~~~~~~~~

Export trained model for inference:

.. code-block:: python

   import torch
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

   # Load trained model
   model = TemporalFusionTransformer.load_from_checkpoint('best_model.ckpt')
   model.eval()

   # Export to TorchScript for production
   example_input = torch.randn(1, 168, 10)  # [batch, seq_len, features]
   scripted_model = torch.jit.script(model)
   scripted_model.save('model_production.pt')

   # Later in production environment
   production_model = torch.jit.load('model_production.pt')

   with torch.no_grad():
       predictions = production_model(new_data)

Batch Prediction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Process large datasets efficiently:

.. code-block:: python

   from transformertf.utils.predict import predict
   from transformertf.data import EncoderDecoderDataModule
   import pandas as pd

   def batch_predict(data_path, model_path, output_path):
       """Process large dataset in batches."""

       # Setup data module for prediction
       data_module = EncoderDecoderDataModule(
           train_df_paths=[data_path],
           target_covariate='target',
           known_covariates=['feature1', 'feature2'],
           ctxt_seq_len=168,
           tgt_seq_len=24,
           batch_size=128,  # Larger batch for efficiency
           normalize=True
       )

       # Generate predictions
       predictions = predict(
           model_ckpt_path=model_path,
           datamodule=data_module
       )

       # Process and save results
       results = []
       for batch_pred in predictions:
           # Convert to pandas DataFrame
           batch_df = pd.DataFrame(batch_pred.numpy())
           results.append(batch_df)

       final_results = pd.concat(results, ignore_index=True)
       final_results.to_parquet(output_path, index=False)

       return final_results

   # Usage
   predictions = batch_predict(
       data_path='large_dataset.parquet',
       model_path='trained_model.ckpt',
       output_path='predictions.parquet'
   )

Performance Optimization Examples
--------------------------------

GPU Memory Optimization
~~~~~~~~~~~~~~~~~~~~~~

Configuration for limited GPU memory:

.. code-block:: yaml

   # memory_optimized.yml
   trainer:
     max_epochs: 100
     accelerator: gpu
     precision: 16                    # Half precision
     gradient_clip_val: 1.0
     accumulate_grad_batches: 4       # Gradient accumulation

   model:
     class_path: transformertf.models.tsmixer.TSMixer
     init_args:
       n_block: 6                     # Smaller model
       ff_dim: 128
       dropout: 0.1

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['data.parquet']
       val_df_paths: ['val.parquet']
       target_covariate: 'target'
       ctxt_seq_len: 100              # Shorter sequences
       tgt_seq_len: 12
       batch_size: 8                  # Smaller batches
       num_workers: 2

Multi-GPU Training
~~~~~~~~~~~~~~~~~

Scale training across multiple GPUs:

.. code-block:: yaml

   # multi_gpu.yml
   trainer:
     max_epochs: 100
     accelerator: gpu
     devices: 4                       # Use 4 GPUs
     strategy: ddp                    # Distributed training
     precision: 16
     sync_batchnorm: true            # Sync batch norm

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       d_model: 256               # Larger model for multi-GPU
       num_heads: 16
       compile_model: true            # PyTorch compilation

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['large_data.parquet']
       val_df_paths: ['large_val.parquet']
       target_covariate: 'target'
       ctxt_seq_len: 336
       tgt_seq_len: 48
       batch_size: 32                 # Per-GPU batch size
       num_workers: 8
       distributed_sampler: true      # Required for DDP

Running Examples
---------------

**CLI Usage:**

.. code-block:: bash

   # Basic training
   transformertf fit --config retail_forecasting.yml

   # With custom experiment name
   transformertf fit --config energy_forecasting.yml --experiment-name energy_v1

   # Prediction
   transformertf predict --config config.yml --ckpt_path checkpoints/best.ckpt

   # Fast development run (1 batch)
   transformertf fit --config config.yml --fast_dev_run 1

**Python API:**

.. code-block:: python

   # Load and modify configuration programmatically
   from lightning.pytorch.cli import LightningCLI
   from transformertf.main import LightningCLI as TransformerTFCLI

   # Custom training loop
   import lightning as L
   from transformertf.data import EncoderDecoderDataModule
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

   # Initialize components
   data_module = EncoderDecoderDataModule(...)
   model = TemporalFusionTransformer(...)

   # Custom callbacks
   callbacks = [
       L.callbacks.ModelCheckpoint(monitor='validation/loss'),
       L.callbacks.EarlyStopping(monitor='validation/loss', patience=15),
       L.callbacks.LearningRateMonitor()
   ]

   trainer = L.Trainer(
       max_epochs=100,
       callbacks=callbacks,
       accelerator='auto'
   )

   trainer.fit(model, data_module)

Tips for Examples
----------------

**Data Preparation:**
- Always split data chronologically for time series
- Include temporal features (hour, day, month)
- Normalize continuous variables
- Handle missing values before training

**Model Selection:**
- Start with simpler models (LSTM) before complex ones (TFT)
- Use TFT for multivariate data with mixed types
- Use TSMixer for fast training on large datasets

**Training:**
- Always use gradient clipping (gradient_clip_val: 1.0)
- Monitor validation loss, not training loss
- Use early stopping to prevent overfitting
- Save multiple checkpoints during training

**Debugging:**
- Use fast_dev_run for quick testing
- Start with small datasets and simple models
- Check data loading with single batch
- Verify model output shapes match expectations
