"""Hyperparameter name migration logic."""

from __future__ import annotations

import re
from pathlib import Path

from .common import (
    MigrationError,
    backup_file,
    load_checkpoint,
    load_yaml_config,
    print_info,
    print_success,
    print_warning,
    save_checkpoint,
    save_yaml_config,
)

# Complete parameter mapping for all consolidated changes
PARAMETER_MAPPINGS = {
    # Model dimension parameters
    "n_dim_model": "d_model",
    "hidden_size": "d_model",  # For LSTM models
    "hidden_dim": "d_model",  # For some transformer variants
    # Attention parameters
    "n_heads": "num_heads",
    "num_attention_heads": "num_heads",
    # Layer count parameters
    "n_layers": "num_layers",
    # Fully connected parameters
    "hidden_dim_fc": "d_fc",
    "n_dim_fc": "d_fc",
    "fc_dim": "d_fc",
    "ff_dim": "d_fc",
    # TFT specific parameters
    "hidden_continuous_dim": "d_hidden_continuous",
    # PETE specific parameters
    "n_enc_heads": "num_encoder_heads",
    "n_dim_selection": "d_selection",
    # TSMixer specific parameters
    "n_block": "num_blocks",
    # Feature count parameters
    "n_features": "num_features",
}


def migrate_checkpoint_hyperparameters(
    checkpoint_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate hyperparameter names in a PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    print_info(f"Processing checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Collect all changes
    changes = []

    # Migrate hyperparameters
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        if isinstance(hparams, dict):
            for old_param, new_param in PARAMETER_MAPPINGS.items():
                if old_param in hparams:
                    if not dry_run:
                        hparams[new_param] = hparams.pop(old_param)
                    changes.append(f"hyper_parameters.{old_param} -> {new_param}")

    # Migrate datamodule hyperparameters
    if "datamodule_hyper_parameters" in checkpoint:
        dm_hparams = checkpoint["datamodule_hyper_parameters"]
        if isinstance(dm_hparams, dict):
            for old_param, new_param in PARAMETER_MAPPINGS.items():
                if old_param in dm_hparams:
                    if not dry_run:
                        dm_hparams[new_param] = dm_hparams.pop(old_param)
                    changes.append(
                        f"datamodule_hyper_parameters.{old_param} -> {new_param}"
                    )

    if not changes:
        print_info(f"No hyperparameter changes needed for {checkpoint_path}")
        return False

    if dry_run:
        print_warning(f"Would apply changes to {checkpoint_path}:")
        for change in changes:
            print(f"  - {change}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(checkpoint_path)
        print_info(f"Created backup: {backup_path}")

    # Save updated checkpoint
    save_checkpoint(checkpoint, checkpoint_path)

    print_success(f"Migrated hyperparameters in {checkpoint_path}")
    for change in changes:
        print(f"  - {change}")
    return True


def migrate_yaml_hyperparameters(
    config_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate hyperparameter names in a YAML configuration file.

    Args:
        config_path: Path to the YAML config file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    print_info(f"Processing YAML config: {config_path}")

    # Load YAML content
    original_content = load_yaml_config(config_path)

    # Apply parameter mappings using regex to preserve formatting
    migrated_content = original_content
    changes = []

    for old_param, new_param in PARAMETER_MAPPINGS.items():
        # Match parameter names followed by colon (YAML key pattern)
        pattern = rf"^(\s*){re.escape(old_param)}(\s*):(.*)$"
        replacement = rf"\1{new_param}\2:\3"

        # Find all matches before replacing
        matches = re.findall(pattern, migrated_content, re.MULTILINE)
        if matches:
            migrated_content = re.sub(
                pattern, replacement, migrated_content, flags=re.MULTILINE
            )
            changes.append(f"{old_param} -> {new_param}")

    if not changes:
        print_info(f"No hyperparameter changes needed for {config_path}")
        return False

    if dry_run:
        print_warning(f"Would apply changes to {config_path}:")
        for change in changes:
            print(f"  - {change}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(config_path)
        print_info(f"Created backup: {backup_path}")

    # Write updated config
    save_yaml_config(migrated_content, config_path)

    print_success(f"Migrated hyperparameters in {config_path}")
    for change in changes:
        print(f"  - {change}")
    return True


def migrate_file_hyperparameters(
    file_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate hyperparameter names in a file (checkpoint or YAML config).

    Args:
        file_path: Path to the file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    if file_path.suffix.lower() in {".ckpt", ".pth", ".pt"}:
        return migrate_checkpoint_hyperparameters(
            file_path, dry_run=dry_run, backup=backup
        )
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        return migrate_yaml_hyperparameters(file_path, dry_run=dry_run, backup=backup)
    msg = (
        f"Unsupported file type: {file_path.suffix}. "
        "Supported types: .ckpt, .pth, .pt, .yaml, .yml"
    )
    raise MigrationError(msg)
