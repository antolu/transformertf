FAQ and Troubleshooting
=======================

Common Questions
----------------

**Q: Which model should I use for my time series data?**

A: Choose based on your data complexity and requirements:

- **LSTM**: Simple univariate or low-dimensional data, need fast training
- **TFT**: Complex multivariate data with mixed categorical/continuous features, need interpretability
- **TSMixer**: Large datasets where training speed is important
- **Transformer**: Standard multivariate forecasting, want attention mechanisms

**Q: How much historical data do I need?**

A: General guidelines:
- Minimum: 10x your prediction horizon (predict 24h → need 240h history)
- Recommended: Include at least 2-3 seasonal cycles
- For daily data predicting weeks ahead: 1-2 years
- For hourly data predicting days ahead: 2-3 months

**Q: Should I normalize my data?**

A: Yes, always set ``normalize: true`` in your data configuration. TransformerTF handles normalization automatically and ensures proper denormalization during prediction.

**Q: How do I handle missing values?**

A: TransformerTF doesn't handle missing values automatically. Fill gaps before training:

.. code-block:: python

   # Forward fill for continuous variables
   df['temperature'] = df['temperature'].fillna(method='ffill')

   # Interpolation for numeric time series
   df['target'] = df['target'].interpolate(method='time')

   # Mode for categorical variables
   df['category'] = df['category'].fillna(df['category'].mode()[0])

**Q: Can I use multiple GPUs?**

A: Yes, configure multi-GPU training:

.. code-block:: yaml

   trainer:
     accelerator: gpu
     devices: 2              # Number of GPUs
     strategy: ddp           # Distributed training

**Q: How do I save and load models?**

A: Lightning automatically saves checkpoints. Load them like this:

.. code-block:: python

   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

   # Load model from checkpoint
   model = TemporalFusionTransformer.load_from_checkpoint('path/to/checkpoint.ckpt')

Common Errors and Solutions
--------------------------

Memory Errors
~~~~~~~~~~~~~

**Error**: ``CUDA out of memory`` or ``RuntimeError: out of memory``

**Solutions**:

1. **Reduce batch size**:

   .. code-block:: yaml

      data:
        init_args:
          batch_size: 16    # Try 8, 16 instead of 32, 64

2. **Reduce sequence length**:

   .. code-block:: yaml

      data:
        init_args:
          ctxt_seq_len: 100  # Instead of 168
          tgt_seq_len: 12    # Instead of 24

3. **Use gradient accumulation**:

   .. code-block:: yaml

      trainer:
        accumulate_grad_batches: 4   # Simulates 4x batch size

4. **Enable mixed precision**:

   .. code-block:: yaml

      trainer:
        precision: 16       # Half precision

5. **Reduce model size**:

   .. code-block:: yaml

      model:
        init_args:
          d_model: 32    # Instead of 64 or 128
          num_heads: 4       # Instead of 8

Training Issues
~~~~~~~~~~~~~~

**Error**: ``NaN`` losses during training

**Causes and Solutions**:

1. **Learning rate too high**:

   .. code-block:: yaml

      optimizer:
        init_args:
          lr: 0.0001         # Try lower learning rate

2. **Missing gradient clipping**:

   .. code-block:: yaml

      trainer:
        gradient_clip_val: 1.0   # Always include this

3. **Data contains infinities**:

   .. code-block:: python

      # Check for infinite values
      df.replace([np.inf, -np.inf], np.nan, inplace=True)
      df.dropna(inplace=True)

4. **Unstable normalization**:

   .. code-block:: yaml

      data:
        init_args:
          normalize: true    # Ensure normalization is enabled

**Error**: Model not converging or poor performance

**Solutions**:

1. **Check data leakage**: Ensure no future information in features
2. **Verify chronological splits**: Never use random train/val splits
3. **Increase model capacity**:

   .. code-block:: yaml

      model:
        init_args:
          d_model: 128   # Increase from 64
          num_lstm_layers: 3 # Add more layers

4. **Adjust learning rate schedule**:

   .. code-block:: yaml

      lr_scheduler:
        class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
        init_args:
          patience: 5
          factor: 0.5

Configuration Errors
~~~~~~~~~~~~~~~~~~~

**Error**: ``ValueError: Shape mismatch`` or tensor size errors

**Causes and Solutions**:

1. **Sequence length mismatch**:

   Ensure model and data sequence lengths match:

   .. code-block:: yaml

      # In data module
      data:
        init_args:
          ctxt_seq_len: 168
          tgt_seq_len: 24

      # These are automatically linked to model, but verify:
      model:
        init_args:
          # ctxt_seq_len: 168  # Auto-linked
          # tgt_seq_len: 24    # Auto-linked

2. **Feature count mismatch**:

   Check that all specified features exist in your data:

   .. code-block:: python

      # Verify features exist
      required_features = ['temperature', 'humidity', 'day_of_week']
      missing_features = set(required_features) - set(df.columns)
      if missing_features:
          print(f"Missing features: {missing_features}")

3. **Output dimension mismatch**:

   For quantile regression, ensure output_dim matches:

   .. code-block:: yaml

      model:
        init_args:
          quantiles: [0.1, 0.5, 0.9]  # 3 quantiles
          # output_dim is auto-calculated as len(quantiles) * tgt_seq_len

Data Loading Errors
~~~~~~~~~~~~~~~~~~

**Error**: ``FileNotFoundError`` or ``ParquetFile`` errors

**Solutions**:

1. **Check file paths**:

   .. code-block:: python

      import os
      for path in ['train.parquet', 'val.parquet']:
          if not os.path.exists(path):
              print(f"Missing file: {path}")

2. **Verify Parquet format**:

   .. code-block:: python

      import pandas as pd
      df = pd.read_parquet('train.parquet')
      print(df.head())
      print(df.dtypes)

3. **Check data types**:

   .. code-block:: python

      # Ensure timestamp is datetime
      df['timestamp'] = pd.to_datetime(df['timestamp'])

      # Ensure target is numeric
      df['target'] = pd.to_numeric(df['target'])

**Error**: ``DataLoader`` timeout or hanging

**Solutions**:

1. **Reduce num_workers**:

   .. code-block:: yaml

      data:
        init_args:
          num_workers: 0     # Use 0 for debugging, 2-4 for production

2. **Check data corruption**:

   .. code-block:: python

      # Test data loading manually
      from transformertf.data import EncoderDecoderDataModule

      dm = EncoderDecoderDataModule(...)
      dm.setup('fit')
      batch = next(iter(dm.train_dataloader()))
      print(batch.keys())

Installation Issues
~~~~~~~~~~~~~~~~~~

**Error**: Import errors or package not found

**Solutions**:

1. **Verify installation**:

   .. code-block:: bash

      pip list | grep transformertf
      python -c "import transformertf; print(transformertf.__version__)"

2. **Check Python version**:

   .. code-block:: bash

      python --version  # Should be 3.11+

3. **Install dependencies**:

   .. code-block:: bash

      pip install torch>=2.2
      pip install lightning>=2.2

4. **Development installation**:

   .. code-block:: bash

      pip install -e ".[dev,test]"

Performance Issues
-----------------

Slow Training
~~~~~~~~~~~~

**Symptoms**: Training takes much longer than expected

**Solutions**:

1. **Use GPU acceleration**:

   .. code-block:: yaml

      trainer:
        accelerator: gpu

2. **Optimize data loading**:

   .. code-block:: yaml

      data:
        init_args:
          num_workers: 4      # Parallel data loading
          batch_size: 64      # Larger batches if memory allows

3. **Enable model compilation**:

   .. code-block:: yaml

      model:
        init_args:
          compile_model: true  # PyTorch 2.0+ compilation

4. **Use mixed precision**:

   .. code-block:: yaml

      trainer:
        precision: 16        # Faster training on modern GPUs

5. **Consider TSMixer for speed**:

   .. code-block:: yaml

      model:
        class_path: transformertf.models.tsmixer.TSMixer

Poor Prediction Quality
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: High validation loss, poor forecasting accuracy

**Diagnostic Steps**:

1. **Check data quality**:

   .. code-block:: python

      # Look for patterns, outliers, missing values
      df.describe()
      df.plot()

2. **Verify temporal ordering**:

   .. code-block:: python

      # Ensure data is sorted by timestamp
      df = df.sort_values('timestamp')

3. **Check for data leakage**:

   .. code-block:: python

      # Future information shouldn't be in known_covariates
      # Only use features available at prediction time

4. **Increase model complexity gradually**:

   .. code-block:: yaml

      # Start simple, add complexity
      model:
        init_args:
          d_model: 32     # → 64 → 128
          num_heads: 4        # → 8 → 16

5. **Longer training**:

   .. code-block:: yaml

      trainer:
        max_epochs: 200      # More epochs

      # Add early stopping
      callbacks:
        - class_path: lightning.pytorch.callbacks.EarlyStopping
          init_args:
            patience: 20

Debugging Tips
-------------

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Start small**:

   .. code-block:: bash

      # Test with minimal data
      transformertf fit --config config.yml --fast_dev_run 1

2. **Use small datasets first**:

   .. code-block:: python

      # Test with subset of data
      small_df = df.head(1000)
      small_df.to_parquet('test_data.parquet')

3. **Enable verbose logging**:

   .. code-block:: bash

      transformertf fit --config config.yml -vv

4. **Check intermediate outputs**:

   .. code-block:: python

      # Inspect data loader outputs
      dm = EncoderDecoderDataModule(...)
      dm.setup('fit')

      for batch in dm.train_dataloader():
          print(f"Encoder input shape: {batch['encoder_input'].shape}")
          print(f"Decoder target shape: {batch['decoder_target'].shape}")
          break

Model Inspection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Load model and inspect
   model = TemporalFusionTransformer.load_from_checkpoint('model.ckpt')

   # Check model structure
   print(model)

   # Count parameters
   total_params = sum(p.numel() for p in model.parameters())
   print(f"Total parameters: {total_params:,}")

   # Check for NaN weights
   for name, param in model.named_parameters():
       if torch.isnan(param).any():
           print(f"NaN found in {name}")

Data Validation
~~~~~~~~~~~~~~

.. code-block:: python

   # Comprehensive data validation
   def validate_data(df):
       print(f"Data shape: {df.shape}")
       print(f"Date range: {df.timestamp.min()} to {df.timestamp.max()}")
       print(f"Missing values: {df.isnull().sum().sum()}")
       print(f"Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
       print(f"Duplicate timestamps: {df.timestamp.duplicated().sum()}")

       # Check for constant columns
       numeric_cols = df.select_dtypes(include=[np.number]).columns
       for col in numeric_cols:
           if df[col].nunique() == 1:
               print(f"Warning: {col} has constant values")

   validate_data(train_df)

Getting Help
-----------

When reporting issues, please include:

1. **Configuration file** (without sensitive data)
2. **Error message** (full traceback)
3. **Data shape and types**: ``df.shape``, ``df.dtypes``
4. **Environment info**: Python version, PyTorch version, GPU info
5. **Minimal reproducible example**

**Useful commands for environment info**:

.. code-block:: bash

   # System info
   python --version
   pip list | grep -E "(torch|lightning|transformertf)"
   nvidia-smi  # For GPU info

.. code-block:: python

   # In Python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU count: {torch.cuda.device_count()}")

**Community Resources**:

- Check existing issues in the repository
- Review the :doc:`examples` for similar use cases
- Consult the :doc:`usage` guide for detailed explanations
- Look at :doc:`configuration` for parameter references
