# Migration and Patch Scripts

This directory contains scripts to migrate existing TransformerTF projects and patch compatibility issues across versions.

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

## Migration Scripts

### 1. `migrate_hyperparameter_names_v0.12.py`

**Purpose:** Migrates both PyTorch Lightning checkpoint files (`.ckpt`, `.pth`) and YAML configuration files (`.yaml`, `.yml`) to update hyperparameter names.

**Basic Usage:**
```bash
# Migrate a single checkpoint
python scripts/migrate_hyperparameter_names_v0.12.py model.ckpt

# Migrate a YAML config
python scripts/migrate_hyperparameter_names_v0.12.py config.yml

# Dry run to see what would change
python scripts/migrate_hyperparameter_names_v0.12.py model.ckpt --dry-run

# Migrate entire directory
python scripts/migrate_hyperparameter_names_v0.12.py --directory configs/
```

**Features:**
- Unified script handles both checkpoints and YAML configs
- Automatically detects file type by extension
- Universal parameter mappings across all models
- Supports dry-run mode to preview changes
- Directory migration support

### 2. `migrate_tft_encoder_alignment_v0.13.py`

**Purpose:** Migrates TFT-family model configurations and checkpoints to use explicit `encoder_alignment='right'` for backwards compatibility with the new left-aligned default.

**Supported Models:**
- TemporalFusionTransformer (TFT)
- PFTemporalFusionTransformer (PFTFT)  
- xTFT

**Basic Usage:**
```bash
# Migrate a config file
python scripts/migrate_tft_encoder_alignment_v0.13.py config.yaml

# Migrate a checkpoint
python scripts/migrate_tft_encoder_alignment_v0.13.py model.ckpt

# Migrate entire directory
python scripts/migrate_tft_encoder_alignment_v0.13.py --directory configs/

# Dry run to preview changes
python scripts/migrate_tft_encoder_alignment_v0.13.py --dry-run config.yaml
```

**Features:**
- Detects TFT-family models automatically
- Adds `encoder_alignment='right'` to data module configuration
- Works with both YAML configs and PyTorch Lightning checkpoints
- Skips non-TFT models

## Patch Scripts (Version Compatibility)

### 1. `patch_add_legacy_target_flag_v0.11.py`

**Purpose:** Adds backwards compatibility for checkpoints with the old `num_future_known_covariates` calculation that included the target variable.

**Usage:**
```bash
python scripts/patch_add_legacy_target_flag_v0.11.py checkpoint.ckpt
python scripts/patch_add_legacy_target_flag_v0.11.py checkpoint.ckpt --output checkpoint_patched.ckpt
```

### 2. `patch_time_format_relative_legacy_v0.8.py`

**Purpose:** Patches v0.7 checkpoints for v0.8 compatibility by changing `time_format` from "relative" to "relative_legacy" due to StandardScaler to MaxScaler default change.

**Usage:**
```bash
python scripts/patch_time_format_relative_legacy_v0.8.py checkpoint.ckpt
```

### 3. `patch_maxscaler_attributes_v0.10.py`

**Purpose:** Updates MaxScaler attributes from `data_min_/data_max_` to `min_/max_` for v0.9.2 to v0.10.0 compatibility.

**Usage:**
```bash
python scripts/patch_maxscaler_attributes_v0.10.py checkpoint.ckpt
```

## Migration Strategy

### For Individual Files

1. **Test with dry-run first:**
   ```bash
   python scripts/migrate_hyperparameter_names_v0.12.py your_model.ckpt --dry-run
   ```

2. **Migrate with default behavior:**
   ```bash
   python scripts/migrate_hyperparameter_names_v0.12.py your_model.ckpt
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
   python scripts/migrate_hyperparameter_names_v0.12.py --dry-run --directory /path/to/project
   ```

3. **Run the migration:**
   ```bash
   python scripts/migrate_hyperparameter_names_v0.12.py --directory /path/to/project
   ```

4. **Migrate TFT-family models if needed:**
   ```bash
   python scripts/migrate_tft_encoder_alignment_v0.13.py --directory /path/to/project
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

### Script Fails to Process Files

- Ensure file extensions are correct (`.ckpt`, `.pth` for checkpoints; `.yaml`, `.yml` for configs)
- Check file permissions and that files exist
- Use absolute paths if running from different directories

### YAML Parsing Errors

If configuration file migration fails:
- Ensure your YAML files are valid
- Check for special characters or formatting issues
- The script preserves formatting but requires valid YAML syntax

### Checkpoint Loading Errors

If migrated checkpoints won't load:
- Verify the checkpoint structure wasn't corrupted
- Check that all required parameters are present
- Create backups before migration to restore if needed

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
- [ ] Migrate hyperparameter names with `migrate_hyperparameter_names_v0.12.py`
- [ ] Migrate TFT encoder alignment with `migrate_tft_encoder_alignment_v0.13.py` if using TFT models
- [ ] Apply version-specific patches if needed
- [ ] Test loading migrated checkpoints
- [ ] Verify migrated configs work with new TransformerTF version
- [ ] Update any hardcoded parameter names in your scripts
- [ ] Update documentation and comments in your code
