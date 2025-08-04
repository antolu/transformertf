# Migration Scripts

This directory contains scripts to migrate existing TransformerTF projects to use the new standardized hyperparameter naming conventions.

## Background

TransformerTF has standardized hyperparameter names across all models to improve consistency and align with academic conventions from the original transformer paper:

### New Parameter Naming Conventions

#### Dimension Parameters (d_*)
- `d_model` - Model embedding dimension (previously `n_dim_model`, `hidden_size`, `hidden_dim`)
- `d_hidden` - Hidden layer dimensions
- `d_fc` - Fully connected layer dimensions (previously `fc_dim`, `ff_dim`, `hidden_dim_fc`, `n_dim_fc`)
- `d_hidden_continuous` - Continuous variable embedding dimension (previously `hidden_continuous_dim`)
- `d_selection` - Variable selection dimension (previously `n_dim_selection`)

#### Count Parameters (num_*)
- `num_heads` - Number of attention heads (previously `n_heads`, `num_attention_heads`)
- `num_layers` - Number of layers (previously `n_layers`)
- `num_encoder_layers` - Number of encoder layers
- `num_decoder_layers` - Number of decoder layers
- `num_blocks` - Number of blocks (previously `n_block`)
- `num_features` - Number of input features (previously `n_features`)
- `num_encoder_heads` - Number of encoder attention heads (previously `n_enc_heads`)

#### Model-Specific Parameters
- `num_lstm_layers` - Number of LSTM layers in TFT

These scripts help migrate existing checkpoints and configuration files to use the new parameter names.

## Available Scripts

### 1. `migrate_hyperparameter_names.py`

Migrates PyTorch Lightning checkpoint files (`.ckpt`, `.pth`) to update hyperparameter names in the saved metadata.

**Basic Usage:**
```bash
# Migrate a single checkpoint
python scripts/migrate_hyperparameter_names.py model.ckpt

# Dry run to see what would change
python scripts/migrate_hyperparameter_names.py model.ckpt --dry-run

# Specify output location
python scripts/migrate_hyperparameter_names.py model.ckpt --output migrated_model.ckpt

# Skip backup creation
python scripts/migrate_hyperparameter_names.py model.ckpt --no-backup
```

**Features:**
- Automatically detects model type for context-specific migrations
- Creates backup files by default (`.ckpt.backup`)
- Preserves all checkpoint data, only updates hyperparameter names
- Supports dry-run mode to preview changes

### 2. `migrate_config_files.py`

Migrates YAML configuration files to use the new parameter names.

**Basic Usage:**
```bash
# Migrate a single config file
python scripts/migrate_config_files.py config.yml

# Migrate multiple files
python scripts/migrate_config_files.py *.yml --batch

# In-place modification
python scripts/migrate_config_files.py config.yml --in-place

# Dry run to see changes
python scripts/migrate_config_files.py config.yml --dry-run

# Specify output location
python scripts/migrate_config_files.py config.yml --output migrated_config.yml
```

**Features:**
- Preserves YAML formatting and comments
- Model-specific parameter mapping based on detected model type
- Batch processing for multiple files
- In-place editing option

### 3. `migrate_project.py`

Batch migration script that finds and migrates all checkpoint and configuration files in a project directory.

**Basic Usage:**
```bash
# Migrate entire project
python scripts/migrate_project.py /path/to/project

# Migrate current directory
python scripts/migrate_project.py .

# Dry run to see what would be migrated
python scripts/migrate_project.py /path/to/project --dry-run

# Only migrate checkpoints
python scripts/migrate_project.py /path/to/project --checkpoints-only

# Only migrate config files
python scripts/migrate_project.py /path/to/project --configs-only

# Non-recursive (current directory only)
python scripts/migrate_project.py /path/to/project --no-recursive
```

**Features:**
- Recursively searches directories for files to migrate
- Processes both checkpoint and configuration files
- Excludes already-migrated and backup files
- Comprehensive reporting

## Migration Strategy

### For Individual Files

1. **Test with dry-run first:**
   ```bash
   python scripts/migrate_hyperparameter_names.py your_model.ckpt --dry-run
   ```

2. **Migrate with backup (default):**
   ```bash
   python scripts/migrate_hyperparameter_names.py your_model.ckpt
   ```

3. **Verify the migration worked:**
   - Load the migrated checkpoint in your code
   - Check that parameter names are correct

### For Entire Projects

1. **Backup your project** (recommended):
   ```bash
   cp -r /path/to/project /path/to/project_backup
   ```

2. **Dry run to see what will change:**
   ```bash
   python scripts/migrate_project.py /path/to/project --dry-run
   ```

3. **Run the migration:**
   ```bash
   python scripts/migrate_project.py /path/to/project
   ```

## Parameter Mapping Reference

| Old Parameter | New Parameter | Context |
|---------------|---------------|---------|
| `n_dim_model` | `d_model` | All models |
| `hidden_size` | `d_model` | LSTM models (for main dimension) |
| `hidden_dim` | `d_model` | Some transformer variants |
| `n_heads` | `num_heads` | All attention models |
| `num_attention_heads` | `num_heads` | All attention models |
| `n_layers` | `num_layers` | General layer counts |
| `hidden_continuous_dim` | `d_hidden_continuous` | TFT models |
| `hidden_dim_fc` | `d_fc` | LSTM models |
| `fc_dim` | `d_fc` | Various models |
| `ff_dim` | `d_fc` | Transformer models |
| `n_block` | `num_blocks` | TSMixer models |
| `n_enc_heads` | `num_encoder_heads` | PETE models |
| `n_dim_selection` | `d_selection` | PETE models |
| `n_features` | `num_features` | All models |

## Example Migrations

### Before Migration (Old Parameter Names)
```yaml
model:
  class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
  init_args:
    n_dim_model: 64
    hidden_continuous_dim: 16
    n_heads: 4
    n_layers: 2
```

### After Migration (New Parameter Names)
```yaml
model:
  class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
  init_args:
    d_model: 64
    d_hidden_continuous: 16
    num_heads: 4
    num_lstm_layers: 2
```

## Troubleshooting

### Script Fails to Find Model Type

If the migration script cannot detect your model type:
- The script will use global mappings only
- You may need to manually verify parameter names after migration
- Check that your checkpoint contains the expected hyperparameters

### YAML Parsing Errors

If configuration file migration fails:
- Ensure your YAML files are valid
- Check for special characters or formatting issues
- The script preserves formatting but requires valid YAML syntax

### Checkpoint Loading Errors

If migrated checkpoints won't load:
- Verify the checkpoint structure wasn't corrupted
- Check that all required parameters are present
- Use the backup files to restore if needed

### Import Errors

If you get import errors when running the scripts:
```bash
# Make sure you're in the scripts directory
cd scripts

# Or use absolute paths
python /path/to/transformertf/scripts/migrate_hyperparameter_names.py model.ckpt
```

## Dependencies

These scripts require:
- `torch` (for checkpoint loading/saving)
- `pyyaml` (for configuration file parsing)

Install with:
```bash
pip install torch pyyaml
```

Or if you have TransformerTF installed:
```bash
pip install transformertf[dev]
```

## Configuration Examples

### Temporal Fusion Transformer
```yaml
model:
  class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
  init_args:
    d_model: 64
    d_hidden_continuous: 16
    num_heads: 4
    num_lstm_layers: 2
```

### Standard Transformer
```yaml
model:
  class_path: transformertf.models.transformer.VanillaTransformer
  init_args:
    d_model: 128
    num_heads: 8
    num_encoder_layers: 6
    num_decoder_layers: 6
    d_fc: 512
```

### TSMixer
```yaml
model:
  class_path: transformertf.models.tsmixer.TSMixer
  init_args:
    d_model: 32
    num_blocks: 8
    d_fc: 256
```

### LSTM
```yaml
model:
  class_path: transformertf.models.lstm.LSTM
  init_args:
    d_model: 64
    num_layers: 2
    d_fc: 128
```

## Support

If you encounter issues with migration:
1. Check that you have the latest version of these scripts
2. Verify your checkpoints and configs are from supported TransformerTF versions
3. Use the `--dry-run` option to diagnose issues without modifying files
4. Keep backups of your original files
5. Report issues at the project repository

## Migration Checklist

- [ ] Backup your project directory
- [ ] Run dry-run migration to preview changes
- [ ] Migrate checkpoint files
- [ ] Migrate configuration files
- [ ] Test loading migrated checkpoints
- [ ] Verify migrated configs work with new TransformerTF version
- [ ] Update any hardcoded parameter names in your scripts
- [ ] Update documentation and comments in your code
