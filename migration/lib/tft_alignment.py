"""TFT encoder alignment migration logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .common import (
    MigrationError,
    backup_file,
    load_checkpoint,
    print_info,
    print_success,
    print_warning,
    save_checkpoint,
)

# TFT-family model class names that require encoder_alignment='right'
TFT_FAMILY_MODEL_CLASSES = {
    "TemporalFusionTransformer",
    "TFT",
    "PFTemporalFusionTransformer",
    "xTFT",
}


def is_tft_family_config(config: dict[str, Any]) -> bool:
    """Check if configuration is for a TFT-family model (TFT, PFTFT, xTFT)."""
    model_config = config.get("model", {})
    class_path = model_config.get("class_path", "")

    # Check class_path for TFT-family models
    for tft_class in TFT_FAMILY_MODEL_CLASSES:
        if tft_class in class_path:
            return True

    # Check init_args for legacy configurations
    init_args = model_config.get("init_args", {})
    model_type = init_args.get("model_type", "")
    return any(name.lower() in model_type.lower() for name in ["tft", "pftft", "xtft"])


def is_tft_family_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """Check if checkpoint is for a TFT-family model."""
    # Check hyper_parameters for model class information
    hparams = checkpoint.get("hyper_parameters", {})
    if isinstance(hparams, dict):
        # Look for class path in hyperparameters
        class_path = hparams.get("_target_", "") or hparams.get("class_path", "")
        for tft_class in TFT_FAMILY_MODEL_CLASSES:
            if tft_class in class_path:
                return True

    # Check datamodule_hyper_parameters for additional hints
    dm_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    if isinstance(dm_hparams, dict) and any(
        param in dm_hparams
        for param in ["d_hidden_continuous", "hidden_continuous_dim"]
    ):
        return True

    # Check model state dict keys for TFT-specific patterns
    state_dict = checkpoint.get("state_dict", {})
    if isinstance(state_dict, dict):
        # Standard TFT patterns
        tft_indicators = [
            "model.variable_selection_encoder",
            "model.variable_selection_decoder",
            "model.static_covariate_encoder",
            "model.lstm_encoder",
            "model.lstm_decoder",
            "model.selection_layer",
        ]
        # PFTemporalFusionTransformer patterns
        pftft_indicators = [
            "model.enc_vs",
            "model.dec_vs",
            "model.static_vs",
            "model.enc_lstm",
            "model.dec_lstm",
            "model.static_enrichment",
        ]
        all_indicators = tft_indicators + pftft_indicators
        if any(
            any(indicator in key for key in state_dict) for indicator in all_indicators
        ):
            return True

    return False


def migrate_yaml_tft_alignment(
    config_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate TFT encoder alignment in a YAML configuration file.

    Args:
        config_path: Path to the YAML config file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    print_info(f"Processing YAML config: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        msg = f"Failed to load YAML config {config_path}: {e}"
        raise MigrationError(msg) from e

    if not is_tft_family_config(config):
        print_info(f"Not a TFT-family config, skipping: {config_path}")
        return False

    # Check if data module configuration exists
    data_config = config.get("data", {})
    if not data_config:
        print_warning(f"No data configuration found in {config_path}")
        return False

    # Add encoder_alignment='right' to data module init_args
    data_init_args = data_config.setdefault("init_args", {})

    migration_action = ""

    if "encoder_alignment" in data_init_args:
        current_alignment = data_init_args["encoder_alignment"]
        if current_alignment != "right":
            migration_action = (
                f"Change encoder_alignment from '{current_alignment}' to 'right'"
            )
            if not dry_run:
                data_init_args["encoder_alignment"] = "right"
        else:
            print_info(f"Config already has encoder_alignment='right': {config_path}")
            return False
    else:
        migration_action = "Add encoder_alignment='right'"
        if not dry_run:
            data_init_args["encoder_alignment"] = "right"

    if dry_run:
        print_warning(f"Would apply TFT alignment changes to {config_path}:")
        print(f"  - {migration_action}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(config_path)
        print_info(f"Created backup: {backup_path}")

    # Save updated config
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    except Exception as e:
        msg = f"Failed to save YAML config {config_path}: {e}"
        raise MigrationError(msg) from e

    print_success(f"Migrated TFT alignment in {config_path}")
    print(f"  - {migration_action}")
    return True


def migrate_checkpoint_tft_alignment(
    checkpoint_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate TFT encoder alignment in a PyTorch Lightning checkpoint.

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

    if not is_tft_family_checkpoint(checkpoint):
        print_info(f"Not a TFT-family checkpoint, skipping: {checkpoint_path}")
        return False

    # Check datamodule_hyper_parameters for encoder_alignment
    dm_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    if not isinstance(dm_hparams, dict):
        print_warning(
            f"No datamodule_hyper_parameters found in checkpoint {checkpoint_path}"
        )
        return False

    migration_action = ""

    if "encoder_alignment" in dm_hparams:
        current_alignment = dm_hparams["encoder_alignment"]
        if current_alignment != "right":
            migration_action = (
                f"Change encoder_alignment from '{current_alignment}' to 'right'"
            )
            if not dry_run:
                dm_hparams["encoder_alignment"] = "right"
        else:
            print_info(
                f"Checkpoint already has encoder_alignment='right': {checkpoint_path}"
            )
            return False
    else:
        migration_action = "Add encoder_alignment='right'"
        if not dry_run:
            dm_hparams["encoder_alignment"] = "right"

    if dry_run:
        print_warning(f"Would apply TFT alignment changes to {checkpoint_path}:")
        print(f"  - {migration_action}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(checkpoint_path)
        print_info(f"Created backup: {backup_path}")

    # Save updated checkpoint
    save_checkpoint(checkpoint, checkpoint_path)

    print_success(f"Migrated TFT alignment in {checkpoint_path}")
    print(f"  - {migration_action}")
    return True


def migrate_file_tft_alignment(
    file_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate TFT encoder alignment in a file (checkpoint or YAML config).

    Args:
        file_path: Path to the file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    if file_path.suffix.lower() in {".ckpt", ".pth", ".pt"}:
        return migrate_checkpoint_tft_alignment(
            file_path, dry_run=dry_run, backup=backup
        )
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        return migrate_yaml_tft_alignment(file_path, dry_run=dry_run, backup=backup)
    msg = (
        f"Unsupported file type: {file_path.suffix}. "
        "Supported types: .ckpt, .pth, .pt, .yaml, .yml"
    )
    raise MigrationError(msg)
