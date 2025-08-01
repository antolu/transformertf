Development Guidelines
=====================

This document outlines development guidelines and conventions for contributing to TransformerTF.

Hyperparameter Naming Conventions
---------------------------------

TransformerTF uses standardized hyperparameter naming conventions across all models and neural network components. This ensures API consistency and follows established transformer architecture standards.

Standard Parameter Names
~~~~~~~~~~~~~~~~~~~~~~~~

**Model Dimension Parameters:**
  - ``d_model``: The primary model dimension (embedding/hidden dimension)
  - Used for: input embeddings, hidden states, attention computations

**Attention Parameters:**
  - ``num_heads``: Number of attention heads in multi-head attention
  - Must be a divisor of ``d_model`` (i.e., ``d_model % num_heads == 0``)

**Layer Count Parameters:**
  - ``num_layers``: General number of layers
  - ``num_encoder_layers``: Specific encoder layer count (for encoder-decoder models)
  - ``num_decoder_layers``: Specific decoder layer count (for encoder-decoder models)

**Context-Specific Parameters:**
  - ``hidden_dim``: Used for internal dimensions in components like MLPs, GRNs
  - Kept distinct from ``d_model`` when referring to intermediate processing dimensions

Motivation for These Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The choice of these parameter names is based on several factors:

**1. Academic Standard Alignment**
  ``d_model`` is the canonical parameter name from the original "Attention Is All You Need" paper (Vaswani et al., 2017). This ensures consistency with academic literature and makes the codebase familiar to researchers.

**2. PyTorch Native Compatibility**
  PyTorch's ``torch.nn.Transformer`` uses ``d_model`` and ``num_encoder_layers``/``num_decoder_layers``. Our conventions align with PyTorch's native implementation.

**3. API Clarity and Consistency**
  - ``num_layers`` is more descriptive than abbreviated forms like ``n_layers``
  - ``num_heads`` clearly indicates the parameter's purpose
  - Full word naming follows modern Python API design principles

**4. Framework Considerations**
  While HuggingFace Transformers uses inconsistent naming (``hidden_size``, ``n_layer``, ``num_hidden_layers``), we chose the most principled approach that prioritizes clarity and academic alignment over framework-specific compatibility.

**5. Mathematical Notation Consistency**
  The ``d_model`` notation directly corresponds to mathematical descriptions in transformer literature, making the code more accessible to researchers and practitioners familiar with the field.

Migration from Legacy Names
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous versions of TransformerTF used inconsistent parameter names:
  - ``n_dim_model`` → ``d_model``
  - ``hidden_size`` → ``d_model`` (when referring to model dimension)
  - ``n_heads`` → ``num_heads``
  - ``num_attention_heads`` → ``num_heads``

This was a breaking change implemented to improve code maintainability and user experience. The standardization ensures that:
  - All transformer models have consistent APIs
  - Parameter meanings are immediately clear from their names
  - The codebase follows established conventions from the transformer literature

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    # Correct parameter usage
    from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
    from transformertf.nn import InterpretableMultiHeadAttention

    # Model instantiation with standard parameters
    model = TemporalFusionTransformer(
        num_past_features=10,
        num_future_features=5,
        ctxt_seq_len=168,
        tgt_seq_len=24,
        d_model=256,        # Model dimension
        num_heads=8,        # Attention heads
        num_lstm_layers=2,  # Layer count
        dropout=0.1
    )

    # Neural network component with standard parameters
    attention = InterpretableMultiHeadAttention(
        d_model=256,        # Must match model dimension
        num_heads=8,        # Must divide d_model evenly
        dropout=0.1
    )

Configuration Files
~~~~~~~~~~~~~~~~~~~

Sample configuration files in ``sample_configs/`` demonstrate the correct parameter usage:

.. code-block:: yaml

    model:
      class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
      init_args:
        d_model: 300           # Model dimension
        num_heads: 4           # Attention heads
        num_lstm_layers: 2     # Number of LSTM layers
        hidden_continuous_dim: 8  # Context-specific dimension
        dropout: 0.1

These conventions ensure that TransformerTF provides a clean, consistent, and academically-aligned API for time series modeling with transformer architectures.
