.. _API_docs:

TransformerTF API documentation
===========================

.. rubric:: Modules

.. autosummary::
   :toctree: api

   .. Add the sub-packages that you wish to document below

    transformertf
    transformertf.data
    transformertf.data.dataset
    transformertf.data.transform
    transformertf.models
    transformertf.models.bwlstm
    transformertf.models.bwlstm.typing
    transformertf.models.lstm
    transformertf.models.phytsmixer
    transformertf.models.temporal_fusion_transformer
    transformertf.models.transformer
    transformertf.models.transformer_v2
    transformertf.models.transformerencoder
    transformertf.models.tsmixer
    transformertf.models.tft
    transformertf.nn
    transformertf.nn.functional
    transformertf.utils
    transformertf.utils.chain_schedulers
    transformertf.utils.predict
    transformertf.main

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
