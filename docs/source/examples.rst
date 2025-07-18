Examples Gallery
================

This gallery showcases practical examples for common time series forecasting scenarios with TransformerTF. Each example includes complete code and configuration files.

Quick Examples
--------------

Simple Univariate Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast a single time series using LSTM:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from transformertf.data import TimeSeriesDataModule
   from transformertf.models.lstm import LSTM
   import lightning as L

   # Generate sample data
   dates = pd.date_range('2022-01-01', periods=1000, freq='D')
   values = np.cumsum(np.random.randn(1000)) + 100
   df = pd.DataFrame({'timestamp': dates, 'value': values})

   # Save train/val splits
   train_df = df.iloc[:800]
   val_df = df.iloc[800:]
   train_df.to_parquet('train.parquet', index=False)
   val_df.to_parquet('val.parquet', index=False)

   # Configure data and model
   data_module = TimeSeriesDataModule(
       train_df_paths=['train.parquet'],
       val_df_paths=['val.parquet'],
       target_covariate='value',
       seq_len=30,
       batch_size=16
   )

   model = LSTM(hidden_size=64, num_layers=2)

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
       target_covariate: 'value'
       seq_len: 30
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
       n_dim_model: 64
       num_heads: 4
       quantiles: [0.1, 0.5, 0.9]
       criterion:
         class_path: transformertf.nn.QuantileLoss

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['sales_train.parquet']
       val_df_paths: ['sales_val.parquet']
       target_covariate: 'sales'
       known_covariates: ['day_of_week', 'month', 'is_holiday']
       ctxt_seq_len: 84    # 12 weeks context
       tgt_seq_len: 14     # 2 weeks prediction
       batch_size: 32

Real-World Scenarios
--------------------

Retail Sales Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~

Multi-location retail sales with seasonal patterns:

.. code-block:: python

   # Data preparation for retail scenario
   import pandas as pd
   import numpy as np

   # Generate multi-store sales data
   np.random.seed(42)
   stores = ['store_A', 'store_B', 'store_C']
   dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

   data = []
   for store in stores:
       # Store-specific base sales
       base_sales = {'store_A': 1000, 'store_B': 1500, 'store_C': 800}[store]

       for date in dates:
           # Seasonal patterns
           day_of_year = date.timetuple().tm_yday
           seasonal = 200 * np.sin(2 * np.pi * day_of_year / 365.25)

           # Weekly pattern
           weekly = 150 * np.sin(2 * np.pi * date.weekday() / 7)

           # Holiday boost (simplified)
           holiday_boost = 300 if date.month == 12 else 0

           # Random noise
           noise = np.random.normal(0, 50)

           sales = base_sales + seasonal + weekly + holiday_boost + noise

           data.append({
               'timestamp': date,
               'store_id': store,
               'sales': max(0, sales),  # No negative sales
               'day_of_week': date.weekday(),
               'month': date.month,
               'day_of_month': date.day,
               'is_weekend': date.weekday() >= 5,
               'is_holiday': date.month == 12 and date.day >= 20
           })

   df = pd.DataFrame(data)

   # Split data chronologically
   train_end = '2022-12-31'
   val_end = '2023-06-30'

   train_df = df[df.timestamp <= train_end]
   val_df = df[(df.timestamp > train_end) & (df.timestamp <= val_end)]
   test_df = df[df.timestamp > val_end]

   # Save datasets
   train_df.to_parquet('retail_train.parquet', index=False)
   val_df.to_parquet('retail_val.parquet', index=False)
   test_df.to_parquet('retail_test.parquet', index=False)

**Configuration:**

.. code-block:: yaml

   # retail_forecasting.yml
   seed_everything: 42

   trainer:
     max_epochs: 150
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       n_dim_model: 128
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
       train_df_paths: ['retail_train.parquet']
       val_df_paths: ['retail_val.parquet']
       target_covariate: 'sales'
       known_covariates:
         - 'day_of_week'
         - 'month'
         - 'day_of_month'
         - 'is_weekend'
         - 'is_holiday'
       static_categorical_variables: ['store_id']
       ctxt_seq_len: 90     # 3 months context
       tgt_seq_len: 30      # 1 month prediction
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

Energy Consumption Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hourly energy demand with weather features:

.. code-block:: yaml

   # energy_forecasting.yml
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
       train_df_paths: ['energy_train.parquet']
       val_df_paths: ['energy_val.parquet']
       target_covariate: 'energy_demand'
       known_covariates:
         - 'temperature'
         - 'humidity'
         - 'hour_of_day'
         - 'day_of_week'
         - 'month'
         - 'is_business_day'
       ctxt_seq_len: 168    # 1 week (hourly data)
       tgt_seq_len: 24      # 1 day ahead
       batch_size: 64
       normalize: true
       num_workers: 4

Stock Price Prediction
~~~~~~~~~~~~~~~~~~~~~

Financial time series with technical indicators:

.. code-block:: yaml

   # stock_prediction.yml
   seed_everything: 42

   trainer:
     max_epochs: 200
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.transformer.VanillaTransformer
     init_args:
       n_dim_model: 256
       num_heads: 8
       num_encoder_layers: 6
       num_decoder_layers: 6
       ff_dim: 1024
       dropout: 0.15

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ['stock_train.parquet']
       val_df_paths: ['stock_val.parquet']
       target_covariate: 'close_price'
       known_covariates:
         - 'volume'
         - 'rsi_14'
         - 'macd'
         - 'moving_avg_50'
         - 'volatility'
       static_categorical_variables: ['sector', 'market_cap_category']
       ctxt_seq_len: 60     # 60 days context
       tgt_seq_len: 5       # 5 days prediction
       batch_size: 16
       normalize: true

Specialized Applications
-----------------------

Physics-Informed Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~

Magnetic field hysteresis modeling for particle accelerators:

.. code-block:: yaml

   # physics_hysteresis.yml
   seed_everything: 42

   trainer:
     max_epochs: 100
     accelerator: auto
     gradient_clip_val: 1.0
     callbacks:
       - class_path: transformertf.callbacks.PlotHysteresisCallback
         init_args:
           plot_every: 10

   model:
     class_path: transformertf.models.bwlstm.BWLSTM1
     init_args:
       hidden_size: 128
       num_layers: 3
       dropout: 0.1

   data:
     class_path: transformertf.data.TimeSeriesDataModule
     init_args:
       train_df_paths: ['magnetic_field_train.parquet']
       val_df_paths: ['magnetic_field_val.parquet']
       target_covariate: 'B_field'
       known_covariates: ['I_current']
       seq_len: 200
       batch_size: 32
       extra_transforms:
         B_field:
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
       n_dim_model: 128
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
       "model.init_args.n_dim_model": {
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
       n_dim_model: 256               # Larger model for multi-GPU
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
