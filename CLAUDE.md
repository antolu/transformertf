# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TransformerTF is a PyTorch Lightning-based framework for time series modeling with transformer architectures. The project focuses on physics applications at CERN, specifically modeling hysteresis in magnetic field transfer functions for particle accelerators.

## Scope and Data Characteristics

- The package is designed to develop transformers for transfer functions
- Does NOT deal with conventional time series with:
  - Seasonality
  - Periodicity
  - Categorical variables like holidays
- Variables are almost always continuous data
- Data is sampled at high rates (1 kHz or more)
- Typically downsampled in the datamodule
- Neither code, tests, nor documentation need to address conventional time series scenarios
- Focuses on sampled data such as:
  - Voltage readings from Hall sensors
  - Measured temperature at sub-second sampling rates

## Common Commands

### Development Environment
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev,test]"
```

### Code Quality
```bash
# Run all linting and formatting checks
ruff check .
ruff format .

# Type checking
mypy transformertf/

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run all tests (parallel by default with -n 4)
pytest

# Run specific test module
pytest tests/models/temporal_fusion_transformer/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=transformertf
```

### Training Models
```bash
# Train using configuration file
transformertf fit --config sample_configs/tft_config.yml

# Train with custom parameters
transformertf fit --model.class_path transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer --data.class_path transformertf.data.EncoderDecoderDataModule

# Transfer learning from checkpoint
transformertf fit --config config.yml --transfer-ckpt path/to/checkpoint.ckpt

# Verbose training
transformertf fit -vv --config config.yml
```

### Model Prediction
```bash
# Run prediction/inference
transformertf predict --config config.yml --ckpt_path checkpoints/best.ckpt
```

## Architecture Overview

### Core Components

1. **Data Pipeline (`transformertf.data/`)**
   - `DataModuleBase`: Base Lightning data module for all time series tasks
   - `EncoderDecoderDataModule`: For sequence-to-sequence modeling
   - `TimeSeriesDataModule`: For basic time series prediction
   - Window generators for creating overlapping time windows
   - Transform system for data preprocessing (normalization, polynomial transforms, etc.)

2. **Models (`transformertf.models/`)**
   - `LightningModuleBase`: Base class for all models with training/validation loops
   - `TransformerModuleBase`: Base for transformer-style architectures
   - Available models: TFT (Temporal Fusion Transformer), LSTM variants, TSMixer, Transformer, PETE
   - Each model has a separate Lightning module in its subdirectory

3. **Neural Network Components (`transformertf.nn/`)**
   - Custom layers: GLU, GRN (Gated Residual Network), Variable Selection
   - Loss functions: QuantileLoss, masked losses
   - Attention mechanisms: InterpretableMultiHeadAttention

4. **Configuration System**
   - Uses Lightning CLI with OmegaConf for YAML-based configuration
   - Sample configs in `sample_configs/` for different model types
   - Configuration links data and model parameters automatically

### Key Patterns

- **Encoder-Decoder Architecture**: Most models follow encoder-decoder pattern for sequence prediction
- **Time Series Samples**: Data is structured as samples with input/target sequences of varying lengths
- **Lightning Integration**: All models inherit from Lightning modules for distributed training
- **Transform Pipeline**: Configurable data transformations applied before model training
- **Hyperparameter Linking**: CLI automatically links data module parameters to model parameters

### Model Types

- **TFT (Temporal Fusion Transformer)**: Primary model for time series forecasting with attention
- **LSTM variants**: Including physics-informed versions (BWLSTM for Bouc-Wen modeling)
- **TSMixer**: MLP-based time series model
- **Transformer**: Standard transformer architecture adapted for time series
- **PETE**: Physics-Enhanced Transformer Encoder

### Data Flow

1. Raw time series data loaded from Parquet files
2. Window generation creates overlapping sequences
3. Transform pipeline applies normalization and feature engineering
4. DataModule creates train/val/test DataLoaders
5. Lightning handles training loop with automatic logging
6. Checkpoints saved with validation metrics for model selection

## Configuration Files

Sample configurations in `sample_configs/` demonstrate:
- Model architecture settings
- Data preprocessing pipelines
- Training hyperparameters
- Transform configurations for physics applications

The CLI system automatically validates configurations and provides helpful error messages for parameter mismatches.

## PyTorch Lightning Design Patterns

- **Hyperparameter Management**:
  - There is no need to pass init arguments in the model LightningModules, nor in the data DataModules, because the init arguments are saved automatically by the `save_hyperparameters` call after `super().__init__()`, and since subclasses call `super().__init__()` first, the subsequent `self.save_hyperparameters()` complements whatever the superclass saved. The init arguments can then be retrieved from the `self.hparams` attribute.
  - All model parameters and data processing parameters must be explicitly in the __init__, and not encapsulated in dataclasses to enable them to be captured by the LightningModule.save_hyperparameters() call.

## Development Best Practices

- Run pre-commit before committing
- Run pytests after each major edit

## Lightning Module Patterns

- Lightning module implementations do not need to implement configure_optimizers, as this is overridden by the LightningCLI

## Naming Conventions

- The lightning module of each model should not be appended with model, but rather the torch.nn.Module that the module wraps should be appended with Model. So for an LSTM model, the LightningModule should be named LSTM, and the wrapped model should be named LSTMModel
- The EncoderDecoderLSTM naming follows this convention: the Lightning module is `EncoderDecoderLSTM`, and the PyTorch model is `EncoderDecoderLSTMModel`

### Hyperparameter Naming

All models use standardized hyperparameter names following PyTorch and academic conventions:

- **`d_model`**: Model dimension (embedding/hidden size) - follows the original "Attention Is All You Need" paper and PyTorch's native transformer implementation
- **`num_heads`**: Number of attention heads - consistent with PyTorch MultiheadAttention
- **`num_layers`**: Number of transformer/encoder/decoder layers - descriptive and clear

**Legacy parameter names are no longer supported:**
- `n_dim_model`, `hidden_size`, `hidden_dim` → use `d_model`
- `n_heads`, `num_attention_heads` → use `num_heads`  
- `n_layers` → use `num_layers`

**Migration:** Use the scripts in `scripts/` directory to migrate existing checkpoints and configuration files from legacy parameter names.

## Initialization Patterns

- `__init__` arguments that are supposed to be passed to a parent module do not need to be explicitly passed by calling the super init, unless explicitly stated in the parent class or if they are torch.nn.Modules, since the save_hyperparameters call will save them to hparams immediately. In short, any `__init__` argument that is not in `save_hyperparameters(ignore=...)` do not need to be passed to super init.

## Merge Request Guidelines

- Merge requests should contain a short summary with 1-2 sentences
- Include sections if applicable:
  - Added
  - Changed
  - Fixed
  - Removed
  - Deprecated
- Add code examples when appropriate, especially for breaking changes
- Clearly call out breaking changes

## GitLab CI/CD Workflow

- Always use GitLab CLI (glab) for creating and editing merge requests
  - Learn how to use glab CLI for efficient merge request management
