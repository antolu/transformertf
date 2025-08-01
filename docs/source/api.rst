.. _API_docs:

TransformerTF API Documentation
===============================

This documentation provides comprehensive API reference for all public classes and functions in the TransformerTF package. All components are fully documented with Numpydoc style docstrings including parameters, examples, and cross-references.

Most Important Classes
======================

Start here for the core components you'll use most frequently:

**Data Modules** - Data loading and preprocessing

- :class:`transformertf.data.EncoderDecoderDataModule` - Multi-step forecasting (most common)
- :class:`transformertf.data.TimeSeriesDataModule` - Sequence-to-sequence modeling
- :class:`transformertf.data.DataModuleBase` - Base class for all data modules

**Models** - Neural network architectures

- :class:`transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer` - Complex multivariate forecasting
- :class:`transformertf.models.lstm.LSTM` - Simple and efficient time series modeling
- :class:`transformertf.models.tsmixer.TSMixer` - Fast MLP-based forecasting
- :class:`transformertf.models.LightningModuleBase` - Base class for all models

**Command Line Interface**

- :func:`transformertf.main.main` - Main CLI entry point
- :class:`transformertf.main.LightningCLI` - Extended Lightning CLI

**Utilities** - Helper functions for common tasks

- :func:`transformertf.utils.predict.predict` - Generate predictions from trained models
- :func:`transformertf.utils.tune.tune` - Hyperparameter optimization with Ray Tune
- :func:`transformertf.utils.configure_optimizers` - Optimizer configuration

Quick Start Example
===================

.. code-block:: python

   from transformertf.data import EncoderDecoderDataModule
   from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
   import lightning as L

   # Setup data
   data_module = EncoderDecoderDataModule(
       train_df_paths=["train.parquet"],
       target_covariate="target",
       ctxt_seq_len=168,
       tgt_seq_len=24
   )

   # Setup model
   model = TemporalFusionTransformer(
       d_model=64,
       num_heads=4
   )

   # Train
   trainer = L.Trainer(max_epochs=100)
   trainer.fit(model, data_module)

Complete API Reference
======================

Main Package
============

.. autosummary::
   :toctree: api

   transformertf
   transformertf.main

Data Module
===========

Data loading, preprocessing, and transformation utilities for time series modeling.

.. autosummary::
   :toctree: api

   transformertf.data
   transformertf.data.datamodule
   transformertf.data.dataset
   transformertf.data.transform

Models Module
=============

Neural network model implementations including base classes and specific architectures.

.. autosummary::
   :toctree: api

   transformertf.models
   transformertf.models.lstm
   transformertf.models.temporal_fusion_transformer
   transformertf.models.transformer
   transformertf.models.transformer_v2
   transformertf.models.tsmixer
   transformertf.models.bwlstm
   transformertf.models.phytsmixer
   transformertf.models.pete
   transformertf.models.gru
   transformertf.models.sa_bwlstm
   transformertf.models.pf_tft
   transformertf.models.xtft
   transformertf.models.xtft_conv
   transformertf.models.transformerxl
   transformertf.models.bwlstm.typing

Neural Network Components
=========================

Neural network layers, loss functions, and building blocks for transformer architectures.

.. autosummary::
   :toctree: api

   transformertf.nn
   transformertf.nn.functional

Utilities
=========

Utility functions for optimization, compilation, prediction, and hyperparameter tuning.

.. autosummary::
   :toctree: api

   transformertf.utils
   transformertf.utils.predict
   transformertf.utils.tune
   transformertf.utils.chain_schedulers

Callbacks
=========

Lightning callbacks for specialized training behavior and monitoring.

.. autosummary::
   :toctree: api

   transformertf.callbacks
