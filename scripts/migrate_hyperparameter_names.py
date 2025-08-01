#!/usr/bin/env python3
"""
Migration script for PyTorch Lightning checkpoint files to update hyperparameter names.

This script migrates checkpoint files (.ckpt, .pth) to use the new standardized
hyperparameter naming conventions introduced in TransformerTF.
"""

from __future__ import annotations

import argparse
import shutil
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

# Context-specific mappings for certain models
CONTEXT_SPECIFIC_MAPPINGS = {
    "TemporalFusionTransformer": {
        "hidden_size": "d_model",
        "hidden_continuous_dim": "d_hidden_continuous",
    },
    "LSTM": {
        "hidden_size": "d_model",
        "hidden_dim_fc": "d_fc",
    },
    "TSMixer": {
        "n_block": "num_blocks",
        "ff_dim": "d_fc",
    },
    "VanillaTransformer": {
        "ff_dim": "d_fc",
    },
}


def detect_model_type(checkpoint_data: dict) -> str | None:
    """Detect the model type from checkpoint data."""
    if "hyper_parameters" not in checkpoint_data:
        return None

    # Try to detect from class path or model structure
    hparams = checkpoint_data["hyper_parameters"]

    # Look for class path indicators
    if isinstance(hparams, dict):
        for value in hparams.values():
            if (
                isinstance(value, str)
                and "temporal_fusion_transformer" in value.lower()
            ):
                return "TemporalFusionTransformer"
            if isinstance(value, str) and "tsmixer" in value.lower():
                return "TSMixer"
            if isinstance(value, str) and "lstm" in value.lower():
                return "LSTM"
            if isinstance(value, str) and "transformer" in value.lower():
                return "VanillaTransformer"

    # Fallback: detect from parameter combinations
    param_keys = set(hparams.keys()) if isinstance(hparams, dict) else set()

    if "hidden_continuous_dim" in param_keys:
        return "TemporalFusionTransformer"
    if "n_block" in param_keys or "ff_dim" in param_keys:
        return "TSMixer"
    if "hidden_dim_fc" in param_keys:
        return "LSTM"

    return None


def migrate_hyperparameters(
    hparams: dict, model_type: str | None = None
) -> tuple[dict, list[str]]:
    """
    Migrate hyperparameter names in a dictionary.

    Returns:
        Tuple of (migrated_hparams, changes_made)
    """
    migrated = hparams.copy()
    changes = []

    # Apply context-specific mappings first
    if model_type and model_type in CONTEXT_SPECIFIC_MAPPINGS:
        for old_param, new_param in CONTEXT_SPECIFIC_MAPPINGS[model_type].items():
            if old_param in migrated:
                migrated[new_param] = migrated.pop(old_param)
                changes.append(f"{old_param} -> {new_param} (context: {model_type})")

    # Apply global mappings for any remaining old parameters
    for old_param, new_param in PARAMETER_MAPPINGS.items():
        if old_param in migrated:
            # Skip if already handled by context-specific mapping
            if model_type and model_type in CONTEXT_SPECIFIC_MAPPINGS:
                if old_param in CONTEXT_SPECIFIC_MAPPINGS[model_type]:
                    continue

            migrated[new_param] = migrated.pop(old_param)
            changes.append(f"{old_param} -> {new_param}")

    return migrated, changes


def migrate_checkpoint(
    input_path: Path,
    output_path: Path | None = None,
    create_backup: bool = True,
    dry_run: bool = False,
) -> bool:
    """
    Migrate a PyTorch Lightning checkpoint file.

    Args:
        input_path: Path to input checkpoint
        output_path: Path for output (defaults to input_path)
        create_backup: Whether to create a backup file
        dry_run: If True, only show what would change without modifying files

    Returns:
        True if migration was successful (or would be successful in dry-run)
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return False

    if output_path is None:
        output_path = input_path

    try:
        # Load checkpoint
        print(f"Loading checkpoint: {input_path}")
        checkpoint = torch.load(input_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            print("Error: Checkpoint is not a dictionary")
            return False

        if "hyper_parameters" not in checkpoint:
            print("Warning: No hyper_parameters found in checkpoint")
            return True

        # Detect model type
        model_type = detect_model_type(checkpoint)
        if model_type:
            print(f"Detected model type: {model_type}")
        else:
            print("Model type not detected, using global mappings only")

        # Migrate hyperparameters
        original_hparams = checkpoint["hyper_parameters"]
        if not isinstance(original_hparams, dict):
            print("Warning: hyper_parameters is not a dictionary")
            return True

        migrated_hparams, changes = migrate_hyperparameters(
            original_hparams, model_type
        )

        if not changes:
            print("No parameter names need migration")
            return True

        print(f"Found {len(changes)} parameters to migrate:")
        for change in changes:
            print(f"  - {change}")

        if dry_run:
            print(f"[DRY RUN] Would save migrated checkpoint to: {output_path}")
            return True

        # Create backup if requested
        if create_backup and output_path == input_path:
            backup_path = input_path.with_suffix(f"{input_path.suffix}.backup")
            print(f"Creating backup: {backup_path}")
            shutil.copy2(input_path, backup_path)

        # Update checkpoint with migrated hyperparameters
        checkpoint["hyper_parameters"] = migrated_hparams

        # Save migrated checkpoint
        print(f"Saving migrated checkpoint: {output_path}")
        torch.save(checkpoint, output_path)

        print("Migration completed successfully!")
        return True

    except Exception as e:
        print(f"Error migrating checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate PyTorch Lightning checkpoints to use new hyperparameter names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single checkpoint (creates backup)
  python migrate_hyperparameter_names.py model.ckpt

  # Dry run to see what would change
  python migrate_hyperparameter_names.py model.ckpt --dry-run

  # Specify output location
  python migrate_hyperparameter_names.py model.ckpt --output migrated_model.ckpt

  # Skip backup creation
  python migrate_hyperparameter_names.py model.ckpt --no-backup
        """,
    )

    parser.add_argument(
        "input_path", type=Path, help="Path to input checkpoint file (.ckpt or .pth)"
    )

    parser.add_argument(
        "--output", "-o", type=Path, help="Output path (defaults to input path)"
    )

    parser.add_argument(
        "--no-backup", action="store_true", help="Don't create backup file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )

    args = parser.parse_args()

    success = migrate_checkpoint(
        input_path=args.input_path,
        output_path=args.output,
        create_backup=not args.no_backup,
        dry_run=args.dry_run,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
