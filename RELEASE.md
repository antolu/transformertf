# Release v0.13.0 - Enhanced Sequence Processing and Model Library Expansion

## Summary
This major release introduces comprehensive RNN packed sequence support with alignment-aware masking, significantly expands the model library with new architectures, and streamlines the import system for better user experience.

## Major New Features

### RNN Packed Sequence Support
Complete infrastructure for efficient variable-length sequence processing:

- Automatic sequence packing that intelligently detects and packs variable-length sequences for optimal LSTM performance
- Full compatibility with PyTorch 2.0 compilation for all packing operations
- Alignment-aware masking supporting both left-aligned (RNN packed) and right-aligned (traditional) sequence masking
- Comprehensive loss masking with all loss functions now supporting masking for variable-length sequences

### New TransformerLSTM Architecture
Advanced hybrid model combining LSTM efficiency with transformer attention mechanisms:

- Dual attention mechanisms with self-attention on encoder outputs and cross-attention between decoder and encoder
- Enhanced sequence modeling providing superior performance over basic LSTM through transformer enhancement
- Full compatibility with efficient sequence packing
- Complete Lightning module with training and inference support

### Simplified Import System
Streamlined package interface for better developer experience:

- Direct model imports from `transformertf.models` for all 20+ available models
- Backwards compatible with existing import patterns
- Better model discoverability through unified interface

## Added

### Models and Architectures
- TransformerLSTM: New hybrid LSTM-Transformer architecture with dual attention mechanisms
- Complete model exports making all models available via `from transformertf.models import ModelName`

### Sequence Processing Infrastructure  
- Sequence packing utilities in `transformertf/utils/sequence.py` with core packing/unpacking and torch.compile support
- Alignment-aware attention masking with enhanced functions supporting both alignment strategies
- Loss function masking for MSELoss, MAELoss, HuberLoss, MAPELoss, and SMAPELoss supporting variable-length sequences

### Migration and Compatibility Tools
- Unified migration scripts in `migration/` directory for handling breaking changes
- Automatic parameter migration through `migrate_hyperparameter_names_v0.12.py` for parameter standardization
- TFT alignment migration using `migrate_tft_encoder_alignment_v0.13.py` for encoder alignment updates
- Version compatibility patches providing scripts for smooth version transitions

### Model Integration
- Full packed sequence support in AttentionLSTM and EncoderDecoderLSTM
- Enhanced collate functions with configurable sequence alignment in the data pipeline
- Tensor-based validation system for encoder alignment requirements

## Changed

### API Standardization
- Loss function signatures standardized to `(y_pred, target, *, mask=None, weights=None)` across all losses
- Parameter naming continues standardization of hyperparameter names such as `d_model` and `num_heads`
- The `mask` and `weights` parameters are now keyword-only for API clarity

### Default Behavior Updates
- DataModule encoder alignment default changed from "right" to "left" for optimal RNN packing
- Loss masking control available through `use_loss_masking` parameter for backward compatibility, defaulting to False for TFT models

### Documentation and Testing
- Enhanced RNN documentation leveraging PyTorch automatic sorting
- Improved TCT training convergence tests for CI reliability
- Comprehensive test coverage for masked loss scenarios and packed sequences

## Removed

- Redundant migration scripts consolidated into unified tools
- Legacy aliases for deprecated function and class aliases cleaned up

## Breaking Changes

### DataModule Encoder Alignment Default Change
The default `encoder_alignment` in DataModules changed from "right" to "left". This change affects TFT, PFTFT, and xTFT models which require explicit alignment specification:

```python
# Required for TFT-family models
datamodule = EncoderDecoderDataModule(
    encoder_alignment="right",  # Must specify explicitly
    decoder_alignment="left"    # Default, optional
)
```

### Loss Function API Changes
Loss function parameter signatures have been standardized:

```python
# Old (deprecated)
loss_fn(y_pred, target, weights, mask)

# New (required)
loss_fn(y_pred, target, mask=mask, weights=weights)
```

### Loss Function Renaming
Loss classes have been renamed for consistency:
- `WeightedMSELoss` becomes `MSELoss`
- `WeightedMAELoss` becomes `MAELoss`
- `WeightedHuberLoss` becomes `HuberLoss`

Deprecated aliases are provided with warnings for the transition period.

## Migration Guide

Migration is only necessary for TFT-based models (TemporalFusionTransformer, PFTemporalFusionTransformer, and xTFT). Other models work with the new defaults without any changes.

### Automated Migration for TFT-based Models

#### Parameter Name Migration
```bash
# Migrate hyperparameter names
python migration/migrate_hyperparameter_names_v0.12.py --directory /path/to/project

# Preview changes first
python migration/migrate_hyperparameter_names_v0.12.py --dry-run config.yaml
```

#### TFT Encoder Alignment Migration
```bash
# Migrate TFT model configurations
python migration/migrate_tft_encoder_alignment_v0.13.py --directory configs/

# Preview changes
python migration/migrate_tft_encoder_alignment_v0.13.py --dry-run config.yaml
```

### Manual Migration for TFT Models

Add explicit encoder alignment to configurations:

```yaml
# config.yml
data:
  class_path: transformertf.data.EncoderDecoderDataModule
  init_args:
    encoder_alignment: "right"  # Required for TFT/PFTFT/xTFT
```

### Import Updates

Take advantage of simplified imports:

```python
# New simplified imports
from transformertf.models import (
    TemporalFusionTransformer,
    TransformerLSTM,
    AttentionLSTM,
    BWLSTM1
)

# Old imports still work
from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
```

## Model Configuration Summary

| Model | Encoder Alignment | Decoder Alignment | Loss Masking | Migration Required |
|-------|------------------|-------------------|--------------|-------------------|
| TransformerLSTM | left | left | True | No (new model) |
| AttentionLSTM | left | left | True | No |
| EncoderDecoderLSTM | left | left | True | No |
| TemporalFusionTransformer | right | left | False | Yes |
| PFTemporalFusionTransformer | right | left | False | Yes |
| xTFT | right | left | False | Yes |
| TemporalConvTransformer | right | left | True | Yes |

## Technical Improvements

### Performance Optimizations
- Intelligent packing detection with automatic optimization based on sequence length variance
- Full torch.compile readiness with selective compilation disable for compatibility
- Memory efficiency through reduced padding overhead via proper sequence packing

### Developer Experience
- Comprehensive documentation with enhanced docstrings and examples throughout
- Better error messages with clear validation and helpful migration guidance
- Simplified debugging through improved logging and tensor shape information

## Available Models

With simplified imports, all models are now easily accessible:

- Base classes: `LightningModuleBase`, `TransformerModuleBase`
- LSTM models: `LSTM`, `GRU`, `AttentionLSTM`, `EncoderDecoderLSTM`, `TransformerLSTM`
- Physics models: `BWLSTM1`, `BWLSTM2`, `BWLSTM3`, `SABWLSTM`
- Transformer models: `TemporalFusionTransformer`, `PFTemporalFusionTransformer`, `xTFT`, `VanillaTransformerV2`, `TemporalConvTransformer`, `PETE`
- Aliases: `TFT`, `TCT`

## Getting Started

### Quick Start with New Features

```python
# Use new TransformerLSTM model
from transformertf.models import TransformerLSTM
from transformertf.data import EncoderDecoderDataModule

# Create model with transformer enhancement
model = TransformerLSTM(
    num_past_features=10,
    num_future_features=3,
    d_model=128,
    num_transformer_blocks=4,
    num_heads=8
)

# Data module with packed sequence support
datamodule = EncoderDecoderDataModule(
    # Standard parameters...
    encoder_alignment="left",  # Optimal for LSTM models
    use_sequence_packing=True  # Enable automatic packing
)
```

### Migration Workflow

The migration process involves the following steps:

1. Back up your project directory
2. Run migrations with `--dry-run` to preview changes
3. Apply migrations by removing the `--dry-run` flag
4. Test configurations to ensure models load correctly
5. Optionally update imports to use the simplified syntax

## Compatibility

- Model behavior for existing TFT models is maintained identically with `use_loss_masking=False`
- All existing import patterns continue to work
- Full compatibility with PyTorch 2.0 compilation
- Migration support includes automated scripts with comprehensive dry-run capabilities

This release significantly enhances TransformerTF's capabilities while providing clear migration paths for existing projects using TFT-based models.
