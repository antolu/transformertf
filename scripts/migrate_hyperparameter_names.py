"""
Migration script for consolidating hyperparameter names in transformertf checkpoints.

This script migrates checkpoints from inconsistent hyperparameter naming to the
standardized convention:
- n_dim_model/hidden_size/hidden_dim -> d_model
- n_heads/num_attention_heads -> num_heads
- n_layers/num_lstm_layers -> num_layers (context-specific)

Usage:
    python scripts/migrate_hyperparameter_names.py checkpoint.ckpt
    python scripts/migrate_hyperparameter_names.py checkpoint.ckpt --output migrated.ckpt
    python scripts/migrate_hyperparameter_names.py checkpoint.ckpt --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import torch

# Standard hyperparameter mappings
GLOBAL_MAPPINGS = {
    # Model dimension mappings
    "n_dim_model": "d_model",
    "hidden_size": "d_model",
    "hidden_dim": "d_model",
    # Attention heads mappings
    "n_heads": "num_heads",
    "num_attention_heads": "num_heads",
    # General layer mappings
    "n_layers": "num_layers",
}

# Model-specific mappings for special cases
MODEL_SPECIFIC_MAPPINGS = {
    "TemporalFusionTransformer": {
        "num_lstm_layers": "num_encoder_layers",  # TFT uses LSTM layers as encoders
    },
    "VanillaTransformer": {
        # Keep separate encoder/decoder layer counts
    },
    "TemporalConvTransformer": {
        "hidden_dim": "d_model",
        "num_attention_heads": "num_heads",
    },
}


def get_model_class_name(checkpoint: dict[str, Any]) -> str | None:
    """Extract model class name from checkpoint."""
    if "hyper_parameters" in checkpoint:
        # Look for _target_ or similar indicators
        hparams = checkpoint["hyper_parameters"]
        if "_target_" in hparams:
            return hparams["_target_"].split(".")[-1]

    # Try to infer from state_dict keys or other indicators
    if "state_dict" in checkpoint:
        state_keys = list(checkpoint["state_dict"].keys())
        if any("temporal_fusion" in key.lower() for key in state_keys):
            return "TemporalFusionTransformer"
        if any("transformer" in key.lower() for key in state_keys):
            return "VanillaTransformer"

    return None


def migrate_hyperparameters(
    hparams: dict[str, Any], model_class: str | None = None
) -> tuple[dict[str, Any], list[str]]:
    """
    Migrate hyperparameters to new naming convention.

    Returns:
        Tuple of (migrated_hparams, list_of_changes)
    """
    migrated = hparams.copy()
    changes = []

    # Apply global mappings
    for old_name, new_name in GLOBAL_MAPPINGS.items():
        if old_name in migrated:
            migrated[new_name] = migrated.pop(old_name)
            changes.append(f"{old_name} -> {new_name}")

    # Apply model-specific mappings
    if model_class and model_class in MODEL_SPECIFIC_MAPPINGS:
        specific_mappings = MODEL_SPECIFIC_MAPPINGS[model_class]
        for old_name, new_name in specific_mappings.items():
            if old_name in migrated:
                migrated[new_name] = migrated.pop(old_name)
                changes.append(f"{old_name} -> {new_name} (model-specific)")

    return migrated, changes


def migrate_checkpoint(
    checkpoint_path: pathlib.Path, dry_run: bool = False
) -> dict[str, Any]:
    """Migrate a single checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    all_changes = []
    model_class = get_model_class_name(checkpoint)

    if model_class:
        print(f"Detected model class: {model_class}")
    else:
        print("Could not detect model class, using global mappings only")

    # Migrate model hyperparameters
    if "hyper_parameters" in checkpoint:
        print("Migrating model hyperparameters...")
        migrated_hparams, changes = migrate_hyperparameters(
            checkpoint["hyper_parameters"], model_class
        )
        if changes:
            all_changes.extend([f"Model: {change}" for change in changes])
            if not dry_run:
                checkpoint["hyper_parameters"] = migrated_hparams
        else:
            print("  No model hyperparameter changes needed")

    # Migrate datamodule hyperparameters
    if "datamodule_hyper_parameters" in checkpoint:
        print("Migrating datamodule hyperparameters...")
        migrated_dm_hparams, changes = migrate_hyperparameters(
            checkpoint["datamodule_hyper_parameters"]
        )
        if changes:
            all_changes.extend([f"DataModule: {change}" for change in changes])
            if not dry_run:
                checkpoint["datamodule_hyper_parameters"] = migrated_dm_hparams
        else:
            print("  No datamodule hyperparameter changes needed")

    # Display all changes
    if all_changes:
        print("\nChanges made:")
        for change in all_changes:
            print(f"  - {change}")
    else:
        print(
            "\nNo hyperparameter changes needed - checkpoint already uses standard names"
        )

    return checkpoint


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint", type=pathlib.Path, help="Path to the checkpoint to migrate"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path for migrated checkpoint. Default: input_name_migrated.ckpt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup of original file (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Don't create backup of original file",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.checkpoint.exists():
        print(
            f"Error: Checkpoint file {args.checkpoint} does not exist", file=sys.stderr
        )
        return 1

    try:
        # Migrate the checkpoint
        migrated_checkpoint = migrate_checkpoint(args.checkpoint, dry_run=args.dry_run)

        if args.dry_run:
            print("\nDry run completed - no files were modified")
            return 0

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            stem = args.checkpoint.stem
            suffix = args.checkpoint.suffix
            output_path = args.checkpoint.with_name(f"{stem}_migrated{suffix}")

        # Create backup if requested
        if args.backup and not args.dry_run:
            backup_path = args.checkpoint.with_name(
                f"{args.checkpoint.stem}_backup{args.checkpoint.suffix}"
            )
            print(f"Creating backup: {backup_path}")
            torch.save(
                torch.load(args.checkpoint, map_location="cpu", weights_only=False),
                backup_path,
            )

        # Save migrated checkpoint
        print(f"Saving migrated checkpoint: {output_path}")
        torch.save(migrated_checkpoint, output_path)

    except Exception as e:
        print(f"Error during migration: {e}", file=sys.stderr)
        return 1
    else:
        print("Migration completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
