#!/usr/bin/env python3
"""
Migration script for PyTorch Lightning checkpoints and YAML configs to update hyperparameter names.

This script migrates both checkpoint files (.ckpt, .pth) and YAML configuration files
to use the new standardized hyperparameter naming conventions introduced in TransformerTF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

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


def migrate_checkpoint(checkpoint_path: Path, dry_run: bool = False) -> bool:
    """Migrate a PyTorch Lightning checkpoint."""
    print(f"Processing checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        print(f"Error: Failed to load {checkpoint_path}")
        return False

    # Migrate hyperparameters
    changes = []
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
        print(f"No changes needed for {checkpoint_path}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would apply changes to {checkpoint_path}:")
        for change in changes:
            print(f"  - {change}")
        return True

    # Save updated checkpoint
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception:
        print(f"Error: Failed to save {checkpoint_path}")
        return False

    print(f"Successfully migrated {checkpoint_path}")
    for change in changes:
        print(f"  - {change}")
    return True


def migrate_yaml_config(config_path: Path, dry_run: bool = False) -> bool:
    """Migrate a YAML configuration file."""
    print(f"Processing YAML config: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            original_content = f.read()
    except Exception:
        print(f"Error: Failed to read {config_path}")
        return False

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
        print(f"No changes needed for {config_path}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would apply changes to {config_path}:")
        for change in changes:
            print(f"  - {change}")
        return True

    # Write updated config
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(migrated_content)
    except Exception:
        print(f"Error: Failed to write {config_path}")
        return False

    print(f"Successfully migrated {config_path}")
    for change in changes:
        print(f"  - {change}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate PyTorch Lightning checkpoints and YAML configs to use new hyperparameter names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single checkpoint
  python migrate_hyperparameter_names_v0.12.py model.ckpt

  # Migrate a YAML config
  python migrate_hyperparameter_names_v0.12.py config.yml

  # Dry run to see what would change
  python migrate_hyperparameter_names_v0.12.py model.ckpt --dry-run

  # Migrate entire directory
  python migrate_hyperparameter_names_v0.12.py --directory configs/
        """,
    )

    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Path to checkpoint, config file, or directory",
    )

    parser.add_argument(
        "--directory", "-d", type=Path, help="Directory to migrate recursively"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )

    args = parser.parse_args()

    if not args.path and not args.directory:
        parser.error("Must specify either a path or --directory")

    target_path = args.directory or args.path

    if not target_path.exists():
        print(f"Error: Path does not exist: {target_path}")
        sys.exit(1)

    migrated_count = 0
    total_count = 0

    if target_path.is_dir():
        # Process directory
        for pattern in ["*.ckpt", "*.pth", "*.yaml", "*.yml"]:
            for file_path in target_path.rglob(pattern):
                total_count += 1
                if (
                    file_path.suffix in {".ckpt", ".pth"}
                    and migrate_checkpoint(file_path, args.dry_run)
                ) or (
                    file_path.suffix in {".yaml", ".yml"}
                    and migrate_yaml_config(file_path, args.dry_run)
                ):
                    migrated_count += 1
    else:
        # Process single file
        total_count = 1
        if target_path.suffix in {".ckpt", ".pth"}:
            if migrate_checkpoint(target_path, args.dry_run):
                migrated_count = 1
        elif target_path.suffix in {".yaml", ".yml"}:
            if migrate_yaml_config(target_path, args.dry_run):
                migrated_count = 1
        else:
            print(f"Error: Unsupported file type: {target_path.suffix}")
            sys.exit(1)

    action = "Would migrate" if args.dry_run else "Migrated"
    print(f"\nMigration complete: {action} {migrated_count}/{total_count} files")
    sys.exit(0 if migrated_count > 0 else 1)


if __name__ == "__main__":
    main()
