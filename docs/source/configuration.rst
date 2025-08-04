Configuration Reference
=======================

TransformerTF uses YAML configuration files with Lightning's CLI system. This reference covers all configuration options and provides templates for common scenarios.

Configuration Structure
-----------------------

All configuration files follow this structure:

.. code-block:: yaml

   # Global training settings
   seed_everything: 42

   # Lightning Trainer configuration
   trainer:
     max_epochs: 100
     accelerator: auto
     # ... trainer parameters

   # Model configuration
   model:
     class_path: transformertf.models.ModelClassName
     init_args:
       # ... model parameters

   # Data module configuration
   data:
     class_path: transformertf.data.DataModuleClassName
     init_args:
       # ... data parameters

   # Optimizer configuration
   optimizer:
     class_path: torch.optim.OptimizerClassName
     init_args:
       # ... optimizer parameters

   # Learning rate scheduler (optional)
   lr_scheduler:
     class_path: torch.optim.lr_scheduler.SchedulerClassName
     init_args:
       # ... scheduler parameters

Core Sections
-------------

Trainer Configuration
~~~~~~~~~~~~~~~~~~~~~

Controls Lightning trainer behavior:

.. code-block:: yaml

   trainer:
     max_epochs: 100                    # Maximum training epochs
     accelerator: auto                  # 'auto', 'cpu', 'gpu', 'tpu'
     devices: auto                      # Number of devices or 'auto'
     precision: 32                      # 16, 32, or 'bf16'
     gradient_clip_val: 1.0            # Gradient clipping threshold
     accumulate_grad_batches: 1        # Gradient accumulation steps
     check_val_every_n_epoch: 1        # Validation frequency
     log_every_n_steps: 50             # Logging frequency
     enable_checkpointing: true        # Enable model checkpointing
     enable_progress_bar: true         # Show progress bars
     deterministic: false              # Deterministic training (slower)

Model Configurations
~~~~~~~~~~~~~~~~~~~~

**Temporal Fusion Transformer (TFT)**

.. code-block:: yaml

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       # Architecture parameters
       d_model: 64                 # Model dimension (32, 64, 128, 256)
       d_hidden_continuous: 16       # Continuous variable processing dim
       num_heads: 4                    # Number of attention heads (4, 8, 16)
       num_lstm_layers: 2              # LSTM encoder/decoder layers (1, 2, 3)
       dropout: 0.1                    # Dropout rate (0.0 - 0.5)

       # Output configuration
       output_dim: 1                   # Output dimension (auto-linked from data)
       quantiles: [0.1, 0.5, 0.9]    # Quantiles for uncertainty (optional)

       # Loss function
       criterion:
         class_path: transformertf.nn.QuantileLoss  # or torch.nn.MSELoss

**LSTM Models**

.. code-block:: yaml

   model:
     class_path: transformertf.models.lstm.LSTM
     init_args:
       d_model: 128               # LSTM hidden dimension
       num_layers: 2                  # Number of LSTM layers
       dropout: 0.1                   # Dropout rate
       bidirectional: false           # Use bidirectional LSTM
       output_dim: 1                  # Output dimension
       quantiles: null                # Optional quantile regression

**TSMixer Models**

.. code-block:: yaml

   model:
     class_path: transformertf.models.tsmixer.TSMixer
     init_args:
       num_blocks: 8                     # Number of mixing blocks
       d_fc: 256                    # Feed-forward dimension
       dropout: 0.1                   # Dropout rate
       activation: "gelu"             # Activation function
       norm_type: "batch_norm"        # Normalization type
       output_dim: 1                  # Output dimension

**Transformer Models**

.. code-block:: yaml

   model:
     class_path: transformertf.models.transformer.VanillaTransformer
     init_args:
       d_model: 128               # Model dimension
       num_heads: 8                   # Attention heads
       num_encoder_layers: 6          # Encoder layers
       num_decoder_layers: 6          # Decoder layers
       d_fc: 512                    # Feed-forward dimension
       dropout: 0.1                   # Dropout rate
       output_dim: 1                  # Output dimension

Data Module Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**EncoderDecoderDataModule (Most Common)**

.. code-block:: yaml

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       # Data paths
       train_df_paths: ["train.parquet"]        # Training data files
       val_df_paths: ["val.parquet"]            # Validation data files
       test_df_paths: ["test.parquet"]          # Test data files (optional)

       # Column specifications
       target_covariate: "magnetic_field"      # Target variable name
       known_covariates:                        # Known future features
         - "voltage_input"
         - "temperature"
         - "frequency"
       static_categorical_variables:            # Entity-level categories
         - "sensor_type"
         - "measurement_location"
       static_real_variables:                   # Entity-level numerics
         - "calibration_factor"
         - "sensor_sensitivity"

       # Sequence configuration
       ctxt_seq_len: 1000                      # Context length (100 seconds at 10 Hz)
       tgt_seq_len: 100                        # Prediction horizon (10 seconds)
       randomize_seq_len: false                # Random sequence lengths
       stride: 1                               # Sampling stride

       # Processing options
       normalize: true                         # Normalize features
       downsample: 1                           # Downsampling factor
       downsample_method: "interval"           # 'interval' or 'average'

       # Training parameters
       batch_size: 32                          # Batch size
       num_workers: 4                          # Data loading workers
       distributed_sampler: false             # Distributed training

**TimeSeriesDataModule**

.. code-block:: yaml

   data:
     class_path: transformertf.data.TimeSeriesDataModule
     init_args:
       train_df_paths: ["sensor_train.parquet"]
       val_df_paths: ["sensor_val.parquet"]
       target_covariate: "magnetic_field"
       known_covariates: ["voltage_input", "temperature"]
       seq_len: 500                            # Fixed sequence length (50 seconds at 10 Hz)
       batch_size: 32
       normalize: true

Optimizer Configurations
~~~~~~~~~~~~~~~~~~~~~~~~

**Adam Optimizer**

.. code-block:: yaml

   optimizer:
     class_path: torch.optim.Adam
     init_args:
       lr: 0.001                      # Learning rate (1e-5 to 1e-2)
       betas: [0.9, 0.999]           # Adam betas
       weight_decay: 1e-4            # L2 regularization
       eps: 1e-8                     # Numerical stability

**AdamW Optimizer**

.. code-block:: yaml

   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.001
       betas: [0.9, 0.999]
       weight_decay: 0.01            # Higher weight decay for AdamW
       eps: 1e-8

**Ranger Optimizer** (Advanced)

.. code-block:: yaml

   optimizer:
     class_path: pytorch_optimizer.Ranger
     init_args:
       lr: 0.001
       alpha: 0.5                    # RAdam parameter
       betas: [0.95, 0.999]         # Different betas
       n_sma_threshold: 5           # SMA threshold
       weight_decay: 1e-4

Learning Rate Schedulers
~~~~~~~~~~~~~~~~~~~~~~~

**ReduceLROnPlateau** (Recommended)

.. code-block:: yaml

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
     init_args:
       mode: min                     # 'min' for loss, 'max' for accuracy
       factor: 0.5                   # LR reduction factor
       patience: 5                   # Epochs to wait
       threshold: 1e-4               # Minimum change threshold
       verbose: true                 # Log LR changes

**StepLR**

.. code-block:: yaml

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.StepLR
     init_args:
       step_size: 10                 # Epochs between reductions
       gamma: 0.1                    # Reduction factor

**CosineAnnealingLR**

.. code-block:: yaml

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.CosineAnnealingLR
     init_args:
       T_max: 50                     # Maximum epochs
       eta_min: 1e-6                 # Minimum learning rate

Configuration Templates
----------------------

Basic Univariate Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple LSTM for single time series:

.. code-block:: yaml

   seed_everything: 42

   trainer:
     max_epochs: 100
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.lstm.LSTM
     init_args:
       d_model: 64
       num_layers: 2
       dropout: 0.1

   data:
     class_path: transformertf.data.TimeSeriesDataModule
     init_args:
       train_df_paths: ["data/train.parquet"]
       val_df_paths: ["data/val.parquet"]
       target_covariate: "value"
       seq_len: 50
       batch_size: 32
       normalize: true

   optimizer:
     class_path: torch.optim.Adam
     init_args:
       lr: 0.001

Complex Multivariate Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TFT with uncertainty quantification:

.. code-block:: yaml

   seed_everything: 42

   trainer:
     max_epochs: 150
     accelerator: auto
     gradient_clip_val: 1.0
     precision: 16                   # Mixed precision training

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       d_model: 128
       d_hidden_continuous: 32
       num_heads: 8
       num_lstm_layers: 2
       dropout: 0.15
       quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
       criterion:
         class_path: transformertf.nn.QuantileLoss

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ["data/train.parquet"]
       val_df_paths: ["data/val.parquet"]
       target_covariate: "demand"
       known_covariates:
         - "temperature"
         - "humidity"
         - "day_of_week"
         - "hour_of_day"
         - "is_holiday"
       static_categorical_variables:
         - "location_id"
         - "store_type"
       ctxt_seq_len: 336              # 2 weeks
       tgt_seq_len: 48                # 2 days
       batch_size: 16
       normalize: true
       num_workers: 4

   optimizer:
     class_path: torch.optim.AdamW
     init_args:
       lr: 0.0005
       weight_decay: 0.01

   lr_scheduler:
     class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
     init_args:
       mode: min
       factor: 0.5
       patience: 10

High-Performance Training
~~~~~~~~~~~~~~~~~~~~~~~~

Optimized for fast training on large datasets:

.. code-block:: yaml

   seed_everything: 42

   trainer:
     max_epochs: 50
     accelerator: gpu
     devices: 2                      # Multi-GPU training
     strategy: ddp                   # Distributed training
     precision: 16                   # Mixed precision
     gradient_clip_val: 1.0
     accumulate_grad_batches: 2      # Gradient accumulation
     sync_batchnorm: true           # Sync batch norm across GPUs

   model:
     class_path: transformertf.models.tsmixer.TSMixer
     init_args:
       num_blocks: 12
       d_fc: 512
       dropout: 0.1
       activation: "gelu"
       compile_model: true            # PyTorch compilation

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ["data/large_train.parquet"]
       val_df_paths: ["data/large_val.parquet"]
       target_covariate: "target"
       known_covariates: ["feat1", "feat2", "feat3"]
       ctxt_seq_len: 168
       tgt_seq_len: 24
       batch_size: 128               # Larger batch size
       num_workers: 8                # More workers
       distributed_sampler: true     # For multi-GPU

   optimizer:
     class_path: pytorch_optimizer.Ranger
     init_args:
       lr: 0.002                     # Higher LR for larger batch

Transfer Learning Template
~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune pre-trained model:

.. code-block:: yaml

   # Add transfer checkpoint path
   transfer_ckpt: "path/to/pretrained_model.ckpt"

   seed_everything: 42

   trainer:
     max_epochs: 30                  # Fewer epochs for fine-tuning
     accelerator: auto
     gradient_clip_val: 1.0

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       # Must match pre-trained model architecture
       d_model: 64
       num_heads: 4
       num_lstm_layers: 2
       # ... other parameters from original model

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths: ["new_domain_train.parquet"]
       val_df_paths: ["new_domain_val.parquet"]
       # Data configuration for new domain
       # Must be compatible with pre-trained model

   optimizer:
     class_path: torch.optim.Adam
     init_args:
       lr: 0.0001                    # Lower LR for fine-tuning

Parameter Guidelines
-------------------

Model Architecture
~~~~~~~~~~~~~~~~~

**Temporal Fusion Transformer:**
- ``d_model``: 32-256, higher for complex data
- ``num_heads``: 4-16, should divide ``d_model``
- ``d_hidden_continuous``: 8-64, typically d_model/2-4
- ``num_lstm_layers``: 1-3, more layers for longer sequences
- ``dropout``: 0.1-0.3, higher for overfitting

**LSTM:**
- ``d_model``: 32-512, scale with data complexity
- ``num_layers``: 1-4, diminishing returns beyond 3
- ``dropout``: 0.1-0.5, apply between layers

**TSMixer:**
- ``num_blocks``: 4-16, more blocks for complex patterns
- ``d_fc``: 128-1024, typically 2-4x model dimension

Data Configuration
~~~~~~~~~~~~~~~~~

**Sequence Lengths:**
- ``ctxt_seq_len``: 1-4x seasonal period (24h, 168h, 8760h)
- ``tgt_seq_len``: 1-50% of context length
- Memory scales quadratically with sequence length

**Batch Size:**
- Start with 32, adjust based on memory and convergence
- Larger batches (64-128) for stable training
- Smaller batches (8-16) for limited memory

**Features:**
- Include temporal features (hour, day, month)
- Normalize continuous variables
- Limit categorical cardinality (<100 unique values)

Training Parameters
~~~~~~~~~~~~~~~~~~

**Learning Rate:**
- Adam: 1e-4 to 1e-2, typically 1e-3
- Larger models need lower LR
- Use scheduling for best results

**Epochs:**
- 50-200 epochs typical
- Use early stopping with patience 10-20
- Monitor validation loss, not training loss

**Gradient Clipping:**
- Always use gradient_clip_val: 1.0
- Prevents gradient explosion in RNNs/Transformers

Common Patterns
--------------

Override Default Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Override specific trainer defaults
   trainer:
     max_epochs: 200
     # Other parameters use Lightning defaults

   # Override specific model parameters
   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       dropout: 0.2
       # Other parameters use model defaults

Multiple Datasets
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       train_df_paths:
         - "data/location_1_train.parquet"
         - "data/location_2_train.parquet"
         - "data/location_3_train.parquet"
       val_df_paths:
         - "data/location_1_val.parquet"
         - "data/location_2_val.parquet"

Custom Transforms
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       # ... other parameters
       extra_transforms:
         temperature:
           - class_path: transformertf.data.transform.LogTransform
         target:
           - class_path: transformertf.data.transform.DiscreteFunctionTransform
             init_args:
               x: "calibration_data.csv"

Validation and Testing
---------------------

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~

TransformerTF automatically validates configurations:

- Parameter linking between data and model
- Required parameters presence
- Value range checks
- Compatibility verification

To validate without training:

.. code-block:: bash

   transformertf fit --config config.yml --fast_dev_run 1

Best Practices
~~~~~~~~~~~~~

1. **Start Simple**: Begin with basic configurations and add complexity
2. **Use Templates**: Modify provided templates rather than writing from scratch
3. **Version Control**: Keep configuration files in version control
4. **Documentation**: Add comments to complex configurations
5. **Validation**: Test configurations with fast_dev_run
6. **Reproducibility**: Always set seed_everything for reproducible results
