.. _usage:

User Guide
==========

This guide covers the essential concepts and common workflows for using TransformerTF effectively. For detailed API documentation, see the :doc:`api` reference.

Getting Started
---------------

Environment Setup
~~~~~~~~~~~~~~~~~

TransformerTF requires Python 3.11+ and PyTorch 2.2+. For best results, use a conda environment:

.. code-block:: bash

   conda create -n transformertf python=3.11
   conda activate transformertf
   pip install transformertf

Data Requirements
~~~~~~~~~~~~~~~~~

TransformerTF expects high-frequency sensor data in Parquet format with specific column structures:

**Required Columns:**
- **Timestamp column**: datetime index for temporal ordering
- **Target variable**: the physical quantity you want to predict
- **Known covariates**: features available at prediction time (e.g., voltage, frequency, temperature)

**Optional Columns:**
- **Static categorical**: sensor-level features (e.g., sensor_type, measurement_location)
- **Static real**: sensor-level numerical features (e.g., calibration_factor, sensor_sensitivity)

**Example Data Structure:**

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Example high-frequency sensor data
   df = pd.DataFrame({
       'timestamp': pd.date_range('2020-01-01', periods=1000, freq='100ms'),  # 10 Hz sampling
       'magnetic_field': np.random.randn(1000).cumsum(),
       'voltage_input': 5 * np.sin(np.arange(1000) * 2 * np.pi / 100),  # 1 Hz sine wave
       'temperature': 20 + 0.1 * np.random.randn(1000),  # Temperature fluctuations
       'sensor_id': 'sensor_1'
   })

   df.to_parquet('sensor_data.parquet')

Core Concepts
-------------

Model Architectures
~~~~~~~~~~~~~~~~~~

**Temporal Fusion Transformer (TFT)**
- Best for: Complex multivariate forecasting with mixed data types
- Features: Variable selection, temporal attention, quantile regression
- Use when: You have multiple time series, mixed static/dynamic features

**LSTM Models**
- Best for: Simple univariate or low-dimensional forecasting
- Features: Efficient training, good for long sequences
- Use when: You have simple time series or limited computational resources

**TSMixer**
- Best for: Fast training on large datasets
- Features: MLP-based mixing of time and feature dimensions
- Use when: You need efficient training with good performance

**Transformer Models**
- Best for: Standard encoder-decoder forecasting
- Features: Attention mechanisms, parallelizable training
- Use when: You want interpretable attention patterns

Data Module Types
~~~~~~~~~~~~~~~~~

**EncoderDecoderDataModule**
- For: Multi-step ahead forecasting (most common)
- Input: Historical context → Future predictions
- Models: TFT, Transformer, TSMixer

**TimeSeriesDataModule**
- For: Sequence-to-sequence prediction
- Input: Sliding windows of equal length
- Models: LSTM, GRU


Common Workflows
----------------

Basic Forecasting
~~~~~~~~~~~~~~~~~

**Step 1: Prepare Configuration**

Create a YAML configuration file:

.. code-block:: yaml

   # config.yml
   seed_everything: true
   trainer:
     max_epochs: 100
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       n_dim_model: 64
       num_heads: 4
       dropout: 0.1

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ["train.parquet"]
       val_df_paths: ["val.parquet"]
       target_covariate: "target"
       known_covariates: ["feature_1", "feature_2"]
       ctxt_seq_len: 168  # 1 week of hourly data
       tgt_seq_len: 24    # Predict next 24 hours
       batch_size: 32

**Step 2: Train Model**

.. code-block:: bash

   transformertf fit --config config.yml

**Step 3: Generate Predictions**

.. code-block:: bash

   transformertf predict --config config.yml --ckpt_path checkpoints/best.ckpt

Multi-Step Ahead Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For forecasting multiple time steps into the future:

.. code-block:: python

   from transformertf.data import EncoderDecoderDataModule
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

   # Configure for multi-step magnetic field prediction
   data_module = EncoderDecoderDataModule(
       train_df_paths=["sensor_data.parquet"],
       target_covariate="magnetic_field",
       known_covariates=["voltage_input", "temperature"],
       ctxt_seq_len=1000,  # Use 100 seconds of context (10 Hz)
       tgt_seq_len=100,    # Predict 10 seconds ahead
       batch_size=16
   )

   model = TemporalFusionTransformer(
       n_dim_model=128,
       hidden_continuous_dim=32,
       num_heads=8,
       output_dim=100     # Match target sequence length
   )

Quantile Regression
~~~~~~~~~~~~~~~~~~~

For uncertainty quantification with prediction intervals:

.. code-block:: yaml

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       quantiles: [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
       criterion:
         class_path: transformertf.nn.QuantileLoss

This produces predictions with confidence bands around the median forecast.

Transfer Learning
~~~~~~~~~~~~~~~~~

Adapt a pre-trained model to new data:

.. code-block:: yaml

   # Add to your configuration
   transfer_ckpt: "path/to/pretrained_model.ckpt"

   # Model architecture should match the checkpoint
   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     # ... same architecture parameters

.. code-block:: bash

   transformertf fit --config config_with_transfer.yml

The model weights are loaded from the checkpoint and fine-tuned on your new dataset.

Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Ray Tune for automated hyperparameter search:

.. code-block:: python

   from transformertf.utils.tune import tune, TuneConfig

   # Define search space
   config = {
       "model.init_args.n_dim_model": {"type": "choice", "values": [32, 64, 128]},
       "model.init_args.num_heads": {"type": "choice", "values": [4, 8]},
       "model.init_args.dropout": {"type": "uniform", "low": 0.1, "high": 0.3},
       "data.init_args.batch_size": {"type": "choice", "values": [16, 32, 64]}
   }

   # Run optimization
   tune_config = TuneConfig(
       base_config="base_config.yml",
       search_space=config,
       num_samples=20,
       max_epochs=50
   )

   best_config = tune(tune_config)

Python API Usage
~~~~~~~~~~~~~~~~

For programmatic control:

.. code-block:: python

   import lightning as L
   from transformertf.data import EncoderDecoderDataModule
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

   # Setup
   data_module = EncoderDecoderDataModule(
       train_df_paths=["data/train.parquet"],
       val_df_paths=["data/val.parquet"],
       target_covariate="magnetic_field",
       known_covariates=["voltage_input", "temperature"],
       ctxt_seq_len=500,  # 50 seconds at 10 Hz
       tgt_seq_len=50     # 5 seconds prediction
   )

   model = TemporalFusionTransformer(
       n_dim_model=64,
       num_heads=4,
       num_lstm_layers=2,
       dropout=0.1
   )

   # Training
   trainer = L.Trainer(
       max_epochs=100,
       accelerator="auto",
       callbacks=[
           L.callbacks.ModelCheckpoint(monitor="validation/loss"),
           L.callbacks.EarlyStopping(monitor="validation/loss", patience=10)
       ]
   )

   trainer.fit(model, data_module)

   # Prediction
   predictions = trainer.predict(model, data_module.test_dataloader())

Working with Configuration Files
---------------------------------

Configuration Structure
~~~~~~~~~~~~~~~~~~~~~~~

TransformerTF uses Lightning's configuration system with automatic parameter linking:

.. code-block:: yaml

   # Top-level training settings
   seed_everything: true
   trainer:
     max_epochs: 100
     accelerator: auto

   # Model configuration
   model:
     class_path: package.module.ClassName
     init_args:
       parameter1: value1
       parameter2: value2

   # Data configuration
   data:
     class_path: package.module.DataModuleClassName
     init_args:
       data_parameter1: value1
       # Sequence lengths are automatically linked to model

Parameter Linking
~~~~~~~~~~~~~~~~~

The framework automatically links data and model parameters to prevent configuration errors:

- ``data.ctxt_seq_len`` → ``model.init_args.ctxt_seq_len``
- ``data.tgt_seq_len`` → ``model.init_args.tgt_seq_len``
- ``data.num_past_known_covariates`` → ``model.init_args.num_past_features``

This ensures your model architecture matches your data configuration.

Common Configuration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For simple univariate forecasting:**

.. code-block:: yaml

   model:
     class_path: transformertf.models.lstm.LSTM
   data:
     class_path: transformertf.data.TimeSeriesDataModule
     init_args:
       seq_len: 50
       target_covariate: "magnetic_field"

**For complex multivariate forecasting:**

.. code-block:: yaml

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       ctxt_seq_len: 1000   # 100 seconds at 10 Hz
       tgt_seq_len: 100     # 10 seconds prediction
       known_covariates: ["voltage_input", "temperature", "frequency"]
       static_categorical_variables: ["sensor_type", "measurement_location"]

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

1. **Normalize your data**: Use ``normalize: true`` in data configuration
2. **Handle missing values**: Fill gaps before training
3. **Feature engineering**: Include temporal features (day of week, hour, etc.)
4. **Validation split**: Use chronological splits, not random

Model Selection
~~~~~~~~~~~~~~~

1. **Start simple**: Try LSTM before more complex models
2. **TFT for complexity**: Use when you have mixed data types and need interpretability
3. **TSMixer for speed**: When you need fast training on large datasets
4. **Batch size**: Start with 32, adjust based on memory and convergence

Training Tips
~~~~~~~~~~~~~

1. **Gradient clipping**: Always use ``gradient_clip_val: 1.0``
2. **Early stopping**: Monitor validation loss with patience
3. **Learning rate**: Use scheduling (StepLR, ReduceLROnPlateau)
4. **Checkpointing**: Save best model based on validation metrics

Troubleshooting
~~~~~~~~~~~~~~~

**Common Issues:**

- **Memory errors**: Reduce batch size or sequence length
- **Poor convergence**: Check learning rate and normalization
- **NaN losses**: Enable gradient clipping and check data for infinities
- **Slow training**: Use appropriate accelerator (GPU) and batch size

**Data Issues:**

- **Shape mismatches**: Verify sequence lengths match between data and model
- **Missing features**: Ensure all specified covariates exist in data
- **Time ordering**: Verify timestamp column is properly sorted
