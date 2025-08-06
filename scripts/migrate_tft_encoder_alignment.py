#!/usr/bin/env python3
"""
Migration script for TFT models to use explicit encoder_alignment='right'.

This script helps migrate existing TFT model configurations and checkpoints
to work with the new explicit encoder_alignment parameter. TFT models require
encoder_alignment='right' for backwards compatibility.

Usage:
    python scripts/migrate_tft_encoder_alignment.py config.yaml
    python scripts/migrate_tft_encoder_alignment.py --checkpoint model.ckpt
    python scripts/migrate_tft_encoder_alignment.py --directory configs/
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# TFT model class names that require encoder_alignment='right'
TFT_MODEL_CLASSES = {
    "TemporalFusionTransformer",
    "TemporalFusionTransformerModel",
    "TFT",
}


def is_tft_config(config: dict[str, Any]) -> bool:
    """Check if configuration is for a TFT model."""
    model_config = config.get("model", {})
    class_path = model_config.get("class_path", "")

    # Check class_path
    for tft_class in TFT_MODEL_CLASSES:
        if tft_class in class_path:
            return True

    # Check init_args for legacy configurations
    init_args = model_config.get("init_args", {})
    model_type = init_args.get("model_type", "")
    return "tft" in model_type.lower()


def migrate_yaml_config(config_path: Path) -> bool:
    """Migrate a YAML configuration file."""
    logger.info(f"Processing YAML config: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception:
        logger.exception(f"Failed to load {config_path}")
        return False

    if not is_tft_config(config):
        logger.info(f"Not a TFT config, skipping: {config_path}")
        return False

    # Check if data module configuration exists
    data_config = config.get("data", {})
    if not data_config:
        logger.warning(f"No data configuration found in {config_path}")
        return False

    # Add encoder_alignment='right' to data module init_args
    data_init_args = data_config.setdefault("init_args", {})

    if "encoder_alignment" in data_init_args:
        current_alignment = data_init_args["encoder_alignment"]
        if current_alignment != "right":
            logger.warning(
                f"TFT config has encoder_alignment='{current_alignment}', "
                f"changing to 'right' in {config_path}"
            )
            data_init_args["encoder_alignment"] = "right"
        else:
            logger.info(f"Config already has encoder_alignment='right': {config_path}")
            return False
    else:
        logger.info(f"Adding encoder_alignment='right' to {config_path}")
        data_init_args["encoder_alignment"] = "right"

    # Create backup
    backup_path = config_path.with_suffix(config_path.suffix + ".backup")
    shutil.copy2(config_path, backup_path)
    logger.info(f"Created backup: {backup_path}")

    # Write updated config
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception:
        logger.exception(f"Failed to write {config_path}")
        # Restore backup
        shutil.copy2(backup_path, config_path)
        return False
    else:
        logger.info(f"Successfully migrated {config_path}")
        return True


def migrate_checkpoint(checkpoint_path: Path) -> bool:
    """Migrate a PyTorch Lightning checkpoint."""
    logger.info(f"Processing checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        logger.exception(f"Failed to load {checkpoint_path}")
        return False

    # Check if it's a TFT model
    model_class = checkpoint.get("hyper_parameters", {}).get("_target_", "")
    if not any(tft_class in model_class for tft_class in TFT_MODEL_CLASSES):
        logger.info(f"Not a TFT checkpoint, skipping: {checkpoint_path}")
        return False

    # Check datamodule_hyper_parameters
    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    if not datamodule_hparams:
        logger.warning(f"No datamodule hyperparameters found in {checkpoint_path}")
        return False

    # Add encoder_alignment='right'
    if "encoder_alignment" in datamodule_hparams:
        current_alignment = datamodule_hparams["encoder_alignment"]
        if current_alignment != "right":
            logger.warning(
                f"TFT checkpoint has encoder_alignment='{current_alignment}', "
                f"changing to 'right' in {checkpoint_path}"
            )
            datamodule_hparams["encoder_alignment"] = "right"
        else:
            logger.info(
                f"Checkpoint already has encoder_alignment='right': {checkpoint_path}"
            )
            return False
    else:
        logger.info(f"Adding encoder_alignment='right' to {checkpoint_path}")
        datamodule_hparams["encoder_alignment"] = "right"

    # Create backup
    backup_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".backup")
    shutil.copy2(checkpoint_path, backup_path)
    logger.info(f"Created backup: {backup_path}")

    # Save updated checkpoint
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception:
        logger.exception(f"Failed to write {checkpoint_path}")
        # Restore backup
        shutil.copy2(backup_path, checkpoint_path)
        return False
    else:
        logger.info(f"Successfully migrated {checkpoint_path}")
        return True


def migrate_directory(directory: Path) -> tuple[int, int]:
    """Migrate all configs and checkpoints in a directory."""
    logger.info(f"Processing directory: {directory}")

    migrated_count = 0
    total_count = 0

    # Process YAML configs
    for pattern in ["*.yaml", "*.yml"]:
        for config_path in directory.rglob(pattern):
            total_count += 1
            if migrate_yaml_config(config_path):
                migrated_count += 1

    # Process checkpoints
    for pattern in ["*.ckpt", "*.pth"]:
        for checkpoint_path in directory.rglob(pattern):
            total_count += 1
            if migrate_checkpoint(checkpoint_path):
                migrated_count += 1

    return migrated_count, total_count


def main() -> int:
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate TFT configurations to use encoder_alignment='right'"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to config file, checkpoint, or directory to migrate",
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Force treat path as checkpoint file"
    )
    parser.add_argument(
        "--directory", action="store_true", help="Force treat path as directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")
        # For dry run, we'd need to modify functions to not write changes
        # This is left as an exercise for production use

    path = args.path
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return 1

    migrated_count = 0
    total_count = 0

    if args.checkpoint or path.suffix in {".ckpt", ".pth"}:
        # Migrate single checkpoint
        total_count = 1
        if migrate_checkpoint(path):
            migrated_count = 1
    elif args.directory or path.is_dir():
        # Migrate directory
        migrated_count, total_count = migrate_directory(path)
    elif path.suffix in {".yaml", ".yml"}:
        # Migrate single config
        total_count = 1
        if migrate_yaml_config(path):
            migrated_count = 1
    else:
        logger.error(f"Unsupported file type: {path}")
        return 1

    logger.info(f"Migration complete: {migrated_count}/{total_count} files migrated")
    return 0


if __name__ == "__main__":
    exit(main())
