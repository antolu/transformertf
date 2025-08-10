# Migration Tool for TransformerTF Projects

This directory contains a unified migration tool for TransformerTF projects, consolidating all migration operations into a single entry point with consistent behavior and interface.

## Quick Start

```bash
# Make the migration tool executable
chmod +x migrate.py

# Run hyperparameter migration
python migrate.py hyperparameters config.yaml model.ckpt

# Run TFT alignment migration
python migrate.py tft-alignment --backup config.yaml

# Apply version upgrade
python migrate.py v0.11-upgrade model.ckpt

# Process entire directory with dry run
python migrate.py hyperparameters --directory ./configs --dry-run
```

## Background

TransformerTF has evolved through several versions with breaking changes requiring migration:

### Parameter Naming Standardization (v0.12)
Standardized hyperparameter names across all models to improve consistency and align with academic conventions:

- **Dimension Parameters**: `d_model`, `d_fc`, `d_hidden_continuous`, `d_selection`
- **Count Parameters**: `num_heads`, `num_layers`, `num_blocks`, `num_features`
- **Model-Specific**: `num_lstm_layers`, `num_encoder_heads`

### TFT Encoder Alignment (v0.13)  
Default encoder alignment changed from "right" to "left". TFT-family models require explicit `encoder_alignment="right"` for compatibility.

### Version-Specific Breaking Changes
- **v0.8**: time_format "relative" → "relative_legacy" (StandardScaler → MaxScaler change)
- **v0.10**: MaxScaler attributes `data_min_/data_max_` → `min_/max_`  
- **v0.11**: Added `_legacy_target_in_future_covariates` flag for backward compatibility

## Commands

### Hyperparameter Migration
```bash
# Migrate parameter names to v0.12 standards
python migrate.py hyperparameters [OPTIONS] FILES...

# Examples
python migrate.py hyperparameters config.yaml
python migrate.py hyperparameters model.ckpt config.yaml
python migrate.py hyperparameters --directory ./configs
python migrate.py hyperparameters --dry-run --backup model.ckpt
```

**Parameter mappings**:
- `n_dim_model`, `hidden_size`, `hidden_dim` → `d_model`
- `n_heads`, `num_attention_heads` → `num_heads`  
- `n_layers` → `num_layers`
- `hidden_continuous_dim` → `d_hidden_continuous`
- `fc_dim`, `ff_dim`, `hidden_dim_fc` → `d_fc`

### TFT Alignment Migration  
```bash
# Add encoder_alignment='right' for TFT-family models
python migrate.py tft-alignment [OPTIONS] FILES...

# Examples
python migrate.py tft-alignment config.yaml
python migrate.py tft-alignment --backup model.ckpt
python migrate.py tft-alignment --directory ./tft_configs
```

**Affected models**: TemporalFusionTransformer, PFTemporalFusionTransformer, xTFT

### Version Upgrades
```bash
# Apply version-specific compatibility fixes
python migrate.py v0.8-upgrade [OPTIONS] CHECKPOINTS...
python migrate.py v0.10-upgrade [OPTIONS] CHECKPOINTS...  
python migrate.py v0.11-upgrade [OPTIONS] CHECKPOINTS...

# Examples
python migrate.py v0.11-upgrade model.ckpt
python migrate.py v0.8-upgrade --backup --directory ./checkpoints
```

**Note**: Version upgrades only work on checkpoint files (.ckpt, .pth, .pt)

## Global Options

All commands support these options:

- `--dry-run`: Preview changes without modifying files
- `--backup`: Create .bak files before modifying (recommended)
- `--directory DIR`: Process all supported files in directory recursively
- `--force`: Skip confirmation prompts
- `-h, --help`: Show help for specific command

## Supported File Types

- **Checkpoints**: `.ckpt`, `.pth`, `.pt` (PyTorch Lightning checkpoints)
- **Configurations**: `.yaml`, `.yml` (YAML configuration files)

## Common Workflows

### Single File Migration
```bash
# Preview what will change
python migrate.py hyperparameters --dry-run model.ckpt

# Apply migration with backup
python migrate.py hyperparameters --backup model.ckpt

# Chain multiple operations (same file paths)
python migrate.py hyperparameters model.ckpt
python migrate.py tft-alignment model.ckpt
python migrate.py v0.11-upgrade model.ckpt
```

### Project-Wide Migration
```bash
# 1. Backup your project first
cp -r /path/to/project /path/to/project_backup

# 2. Preview all changes
python migrate.py hyperparameters --dry-run --directory /path/to/project

# 3. Apply hyperparameter migration
python migrate.py hyperparameters --backup --directory /path/to/project

# 4. Apply TFT alignment if needed
python migrate.py tft-alignment --directory /path/to/project

# 5. Apply version upgrades if needed
python migrate.py v0.11-upgrade --directory /path/to/project
```

### Safe Migration Strategy
```bash
# Always start with dry run
python migrate.py hyperparameters --dry-run config.yaml

# Use backup mode for safety
python migrate.py hyperparameters --backup config.yaml

# Test loading after migration
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Config loads successfully')
"
```

## Migration Examples

### Before Migration
```yaml
model:
  class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
  init_args:
    n_dim_model: 64
    hidden_continuous_dim: 16
    n_heads: 4
    n_layers: 2

data:
  class_path: transformertf.data.EncoderDecoderDataModule
  init_args:
    # encoder_alignment not specified (defaults to "left" in v0.13)
```

### After Migration
```yaml
model:
  class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
  init_args:
    d_model: 64
    d_hidden_continuous: 16
    num_heads: 4
    num_lstm_layers: 2

data:
  class_path: transformertf.data.EncoderDecoderDataModule
  init_args:
    encoder_alignment: "right"  # Explicitly added for TFT compatibility
```

## Troubleshooting

### Common Issues

**"File does not exist" errors**:
- Verify file paths are correct
- Use absolute paths if running from different directories

**"Unsupported file type" errors**:
- Ensure files have correct extensions (.ckpt, .pth, .pt, .yaml, .yml)
- Check file is not corrupted

**YAML parsing errors**:
- Validate YAML syntax before migration
- Check for special characters or formatting issues
- The tool preserves formatting but requires valid YAML

**Checkpoint loading errors**:
- Verify checkpoint isn't corrupted
- Ensure you have required dependencies (torch, pyyaml)

### Recovery

If migration fails or causes issues:

1. **Restore from backup**: `cp config.yaml.bak config.yaml`
2. **Use dry-run mode**: Always test with `--dry-run` first
3. **Process incrementally**: Migrate one file at a time for debugging

### Getting Help
```bash
# General help
python migrate.py --help

# Command-specific help
python migrate.py hyperparameters --help
python migrate.py tft-alignment --help
python migrate.py v0.11-upgrade --help
```

## Dependencies

The migration tool requires:
- `torch` (for checkpoint loading/saving)
- `pyyaml` (for YAML configuration parsing)

Install with:
```bash
pip install torch pyyaml
```

Or if you have TransformerTF installed:
```bash
pip install transformertf[dev]
```

## Legacy Scripts

The old individual migration scripts have been consolidated into this unified tool. The legacy scripts are no longer needed and have been removed:

- ~~`migrate_hyperparameter_names_v0.12.py`~~ → `python migrate.py hyperparameters`
- ~~`migrate_tft_encoder_alignment_v0.13.py`~~ → `python migrate.py tft-alignment`  
- ~~`patch_add_legacy_target_flag_v0.11.py`~~ → `python migrate.py v0.11-upgrade`
- ~~`patch_maxscaler_attributes_v0.10.py`~~ → `python migrate.py v0.10-upgrade`
- ~~`patch_time_format_relative_legacy_v0.8.py`~~ → `python migrate.py v0.8-upgrade`

## Benefits of Unified Tool

- **Consistent interface**: Same arguments work across all migration types
- **No file path changes**: Modify files in-place, making it easy to chain operations
- **Safe defaults**: Optional backup mode prevents accidental data loss  
- **Better error handling**: Clear error messages and validation
- **Simplified workflow**: One tool to remember instead of multiple scripts
