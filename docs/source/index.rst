===================================================
TransformerTF - Transformers for Time Series
===================================================

TransformerTF is a PyTorch Lightning framework for time series forecasting using transformer architectures and other neural networks. Originally developed for physics applications at CERN, it provides production-ready tools for multi-horizon forecasting with uncertainty quantification.

Key Features
------------

- **Multiple Architectures**: Temporal Fusion Transformer (TFT), LSTM variants, TSMixer, standard Transformers
- **Physics-Informed Models**: Specialized models for structural dynamics and magnetic field modeling
- **Flexible Data Pipeline**: Automated preprocessing, windowing, and feature engineering
- **Lightning Integration**: Scalable training with automatic logging, checkpointing, and distributed support
- **Configuration-Driven**: YAML-based workflows for reproducible experiments
- **Uncertainty Quantification**: Built-in quantile regression for prediction intervals

Quick Start
-----------

Install TransformerTF:

.. code-block:: bash

   pip install transformertf

Train a Temporal Fusion Transformer:

.. code-block:: bash

   # Using a sample configuration
   transformertf fit --config sample_configs/tft_config.yml

   # Or with custom data
   transformertf fit \
     --model.class_path transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer \
     --data.class_path transformertf.data.EncoderDecoderDataModule \
     --data.init_args.train_df_paths='["sensor_data.parquet"]' \
     --data.init_args.target_covariate="magnetic_field"

Python API usage:

.. code-block:: python

   from transformertf.data import EncoderDecoderDataModule
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
   import lightning as L

   # Setup data and model
   data_module = EncoderDecoderDataModule(
       train_df_paths=["sensor_data.parquet"],
       target_covariate="magnetic_field",
       ctxt_seq_len=200,
       tgt_seq_len=100,
       batch_size=32
   )

   model = TemporalFusionTransformer(
       d_model=64,
       num_heads=4,
       dropout=0.1
   )

   # Train
   trainer = L.Trainer(max_epochs=100, accelerator="auto")
   trainer.fit(model, data_module)

Core Workflow
-------------

1. **Prepare Data**: Load high-frequency sensor data in Parquet format with physical measurements
2. **Configure Model**: Choose architecture (TFT for complex multivariate, LSTM for simple univariate)
3. **Train**: Use Lightning CLI or Python API with automatic hyperparameter linking
4. **Evaluate**: Built-in metrics and visualization callbacks
5. **Deploy**: Export models for inference with prediction utilities

Installation Options
--------------------

**Basic Installation**:

.. code-block:: bash

   pip install transformertf

**Development Installation**:

.. code-block:: bash

   git clone https://gitlab.cern.ch/dsb/hysteresis/transformertf.git
   cd transformertf
   pip install -e ".[dev,test]"

**With Optional Dependencies**:

.. code-block:: bash

   pip install "transformertf[doc]"  # Documentation tools
   pip install "transformertf[dev]"  # Development tools

Documentation Contents
----------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    self

.. toctree::
    :caption: User Guide
    :maxdepth: 2

    usage
    alignment_docs
    tutorials/index
    examples
    extending_api
    configuration
    evaluation

.. toctree::
    :caption: Reference
    :maxdepth: 1

    api
    faq

.. toctree::
    :caption: Development
    :maxdepth: 1

    development

.. toctree::
    :caption: Index
    :maxdepth: 1

    genindex
