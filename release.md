# Release Notes

New models, architecture improvements, and bug fixes since v0.10.0.

## Breaking Changes

- **Removed `EncoderDataset` and `EncoderDataModule`** - use `EncoderDecoderDataModule` instead
- Renamed `AttentionLSTM` classes to follow naming conventions  
- Renamed `masked_mse_loss` to `mse_loss`

## New

- **AttentionLSTM** model with attention mechanisms
- **EncoderDecoderLSTM** full encoder-decoder LSTM architecture
- **Temporal Convolutional Transformer (TCT)** model
- **ScaledDotProductAttention** standalone attention mechanism
- Standardized column naming with helper functions from `_covariates.py`
- WindowStrategy pattern for decoupling windowing logic
- DatasetFactory integration to reduce code duplication
- `test_step` and `predict_step` methods for LSTM and GRU models
- Configurable `logging_metrics` system
- API extension guide documentation
- Physics-focused signal modeling documentation
- Model evaluation guide

## Changed

- Refactored DataModule architecture
- Standardized dropout instantiation patterns
- Better `__str__` methods for transform classes
- Attention weights only returned when not in training
- Improved error handling in transformer base classes

## Fixed

- Inconsistent `logging_metrics` parameter handling
- Column naming inconsistencies between factory and implementation
- InterpretableMultiHeadAttention identity matrix hack
- Pre-commit hooks and linting issues
- Broken tests after architectural changes
- Redundant attribute assignments

## Removed

- `EncoderDataset` and `EncoderDataModule` classes (~500 lines)
- `DecoderSample` type (merged into `EncoderDecoderSample`)
- `DatasetFactory.create_encoder_dataset()` method
- Redundant logging_metrics assignments
