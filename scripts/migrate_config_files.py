"""
Migration script for updating hyperparameter names in YAML configuration files.

This script updates YAML configuration files to use the standardized hyperparameter
naming convention:
- n_dim_model/hidden_size/hidden_dim -> d_model
- n_heads/num_attention_heads -> num_heads
- n_layers/num_lstm_layers -> num_layers

Usage:
    python scripts/migrate_config_files.py config.yml
    python scripts/migrate_config_files.py config.yml --output migrated_config.yml
    python scripts/migrate_config_files.py config.yml --dry-run
    python scripts/migrate_config_files.py *.yml --batch
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Any

import yaml

# Parameter mapping rules
PARAMETER_MAPPINGS = {
    "n_dim_model": "d_model",
    "n_heads": "num_heads",
    "num_attention_heads": "num_heads",
    "n_layers": "num_layers",
}

# Context-specific mappings for certain model types
MODEL_CONTEXT_MAPPINGS = {
    "LSTM": {
        "hidden_size": "d_model",  # LSTM models use d_model for main dimension
    },
    "AttentionLSTM": {
        "hidden_size": "d_model",
    },
    "TransformerLSTM": {
        "hidden_size": "d_model",
    },
    "TemporalConvTransformer": {
        "hidden_dim": "d_model",
    },
    "PhyTSMixer": {
        "n_dim_model": "d_model",
    },
}


def detect_model_type(config: dict[str, Any]) -> str | None:
    """Detect model type from configuration."""
    if "model" in config and "class_path" in config["model"]:
        class_path = config["model"]["class_path"]
        return class_path.split(".")[-1]
    return None


def migrate_yaml_content(
    content: str, model_type: str | None = None
) -> tuple[str, list[str]]:
    """
    Migrate YAML content by replacing parameter names.

    Args:
        content: Raw YAML content as string
        model_type: Detected model type for context-specific replacements

    Returns:
        Tuple of (migrated_content, list_of_changes)
    """
    migrated_content = content
    changes = []

    # Apply global parameter mappings
    for old_param, new_param in PARAMETER_MAPPINGS.items():
        pattern = rf"(\s+){old_param}(\s*:)"
        replacement = rf"\1{new_param}\2"

        if re.search(pattern, migrated_content):
            migrated_content = re.sub(pattern, replacement, migrated_content)
            changes.append(f"{old_param} -> {new_param}")

    # Apply model-specific mappings
    if model_type and model_type in MODEL_CONTEXT_MAPPINGS:
        specific_mappings = MODEL_CONTEXT_MAPPINGS[model_type]
        for old_param, new_param in specific_mappings.items():
            pattern = rf"(\s+){old_param}(\s*:)"
            replacement = rf"\1{new_param}\2"

            if re.search(pattern, migrated_content):
                migrated_content = re.sub(pattern, replacement, migrated_content)
                changes.append(f"{old_param} -> {new_param} (model-specific)")

    return migrated_content, changes


def migrate_config_file(
    config_path: pathlib.Path, dry_run: bool = False
) -> tuple[str, list[str]]:
    """Migrate a single configuration file."""
    print(f"Processing: {config_path}")

    # Read original content
    with open(config_path, encoding="utf-8") as f:
        original_content = f.read()

    # Parse YAML to detect model type
    try:
        config_data = yaml.safe_load(original_content)
        model_type = detect_model_type(config_data)
        if model_type:
            print(f"  Detected model type: {model_type}")
    except yaml.YAMLError:
        print("  Warning: Could not parse YAML, using global mappings only")
        model_type = None

    # Migrate content
    migrated_content, changes = migrate_yaml_content(original_content, model_type)

    # Display changes
    if changes:
        print("  Changes made:")
        for change in changes:
            print(f"    - {change}")
    else:
        print("  No changes needed")

    return migrated_content, changes


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config_files",
        nargs="+",
        type=pathlib.Path,
        help="Configuration file(s) to migrate",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path (only for single file). Default: input_name_migrated.yml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple files, creating _migrated versions",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup of original files (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Don't create backup of original files",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify files in-place instead of creating new files",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    # Validate arguments
    if not args.batch and len(args.config_files) > 1 and not args.output:
        print("Error: Multiple files require --batch or --output", file=sys.stderr)
        return 1

    if args.output and len(args.config_files) > 1:
        print("Error: --output can only be used with a single file", file=sys.stderr)
        return 1

    successful_migrations = 0
    total_files = len(args.config_files)

    for config_path in args.config_files:
        if not config_path.exists():
            print(f"Error: File {config_path} does not exist", file=sys.stderr)
            continue

        try:
            # Migrate the file
            migrated_content, changes = migrate_config_file(
                config_path, dry_run=args.dry_run
            )

            if not changes:
                print(f"  Skipping {config_path} - no changes needed")
                successful_migrations += 1
                continue

            if args.dry_run:
                print(f"  Dry run: would modify {config_path}")
                continue

            # Determine output path
            if args.in_place:
                output_path = config_path
            elif args.output:
                output_path = args.output
            elif args.batch:
                stem = config_path.stem
                suffix = config_path.suffix
                output_path = config_path.with_name(f"{stem}_migrated{suffix}")
            else:
                stem = config_path.stem
                suffix = config_path.suffix
                output_path = config_path.with_name(f"{stem}_migrated{suffix}")

            # Create backup if requested and not in-place
            if args.backup and not args.in_place:
                backup_path = config_path.with_name(
                    f"{config_path.stem}_backup{config_path.suffix}"
                )
                print(f"  Creating backup: {backup_path}")
                backup_path.write_text(
                    config_path.read_text(encoding="utf-8"), encoding="utf-8"
                )

            # Write migrated content
            print(f"  Writing migrated config: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(migrated_content)

            successful_migrations += 1

        except Exception as e:
            print(f"Error migrating {config_path}: {e}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDry run completed - {total_files} files analyzed")
    else:
        print(
            f"\nMigration completed: {successful_migrations}/{total_files} files migrated successfully"
        )

    return 0 if successful_migrations == total_files else 1


if __name__ == "__main__":
    sys.exit(main())
