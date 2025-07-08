"""
TransformerTF: A PyTorch Lightning Framework for Time Series Modeling.

TransformerTF is a comprehensive toolkit for time series forecasting using state-of-the-art
transformer models and other neural network architectures. Originally developed for modeling
hysteresis in magnetic field transfer functions for particle accelerators at CERN, the
framework supports various time series prediction tasks with configurable data pipelines
and multiple model architectures.

Key Features
------------
- Multiple model architectures including Temporal Fusion Transformer (TFT), LSTM variants,
  TSMixer, standard Transformers, and physics-informed models
- Flexible data pipeline with configurable preprocessing, windowing, and transformation systems
- Built on PyTorch Lightning for scalable training with automatic logging and checkpointing
- Configuration-driven workflow using YAML files for reproducible experiments
- Support for encoder-decoder and sequence-to-sequence modeling patterns

Basic Usage
-----------
Train a model using the command-line interface:

    $ transformertf fit --config sample_configs/tft_config.yml

Or use the Python API:

    >>> from transformertf.data import EncoderDecoderDataModule
    >>> from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
    >>> import lightning as L
    >>>
    >>> # Initialize components
    >>> data_module = EncoderDecoderDataModule(
    ...     train_df_paths=["train.parquet"],
    ...     target_covariate="target",
    ...     ctxt_seq_len=200,
    ...     tgt_seq_len=100
    ... )
    >>> model = TemporalFusionTransformer(n_dim_model=32, num_heads=4)
    >>>
    >>> # Train
    >>> trainer = L.Trainer(max_epochs=100)
    >>> trainer.fit(model, data_module)

Modules
-------
data : transformertf.data
    Data loading, preprocessing, and transformation utilities
models : transformertf.models
    Neural network model implementations
nn : transformertf.nn
    Neural network layers and loss functions
utils : transformertf.utils
    Utility functions for optimization, compilation, and prediction
callbacks : transformertf.callbacks
    Lightning callbacks for specialized training behavior
main : transformertf.main
    Command-line interface and training orchestration

See Also
--------
transformertf.main.main : Main entry point for CLI
transformertf.data.DataModuleBase : Base class for data modules
transformertf.models.LightningModuleBase : Base class for models
"""

from ._version import version as __version__  # noqa
