#!/usr/bin/env python3
"""
Migration script for YAML configuration files to update hyperparameter names.

This script migrates YAML configuration files to use the new standardized
hyperparameter naming conventions introduced in TransformerTF.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


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

# Context-specific mappings based on model class path
CONTEXT_SPECIFIC_MAPPINGS = {
    "temporal_fusion_transformer": {
        "hidden_size": "d_model",
        "hidden_continuous_dim": "d_hidden_continuous",
    },
    "lstm": {
        "hidden_size": "d_model",
        "hidden_dim_fc": "d_fc",
    },
    "tsmixer": {
        "n_block": "num_blocks",
        "ff_dim": "d_fc",
    },
    "transformer": {
        "ff_dim": "d_fc",
    },
}


def detect_model_type_from_config(config_data: dict) -> str | None:
    """Detect the model type from configuration data."""
    if not isinstance(config_data, dict):
        return None

    # Look for model class path
    model_config = config_data.get("model", {})
    if isinstance(model_config, dict):
        class_path = model_config.get("class_path", "")
        if isinstance(class_path, str):
            class_path_lower = class_path.lower()
            if "temporal_fusion_transformer" in class_path_lower:
                return "temporal_fusion_transformer"
            if "tsmixer" in class_path_lower:
                return "tsmixer"
            if "lstm" in class_path_lower:
                return "lstm"
            if "transformer" in class_path_lower:
                return "transformer"

    return None


def migrate_yaml_content(
    yaml_content: str, model_type: str | None = None
) -> tuple[str, list[str]]:
    """
    Migrate parameter names in YAML content while preserving formatting.

    Args:
        yaml_content: Original YAML content as string
        model_type: Detected model type for context-specific mappings

    Returns:
        Tuple of (migrated_content, changes_made)
    """
    migrated_content = yaml_content
    changes = []

    # Get the appropriate mappings
    mappings_to_apply = PARAMETER_MAPPINGS.copy()

    # Apply context-specific mappings if model type is detected
    if model_type and model_type in CONTEXT_SPECIFIC_MAPPINGS:
        mappings_to_apply.update(CONTEXT_SPECIFIC_MAPPINGS[model_type])

    # Apply parameter mappings using regex to preserve formatting
    for old_param, new_param in mappings_to_apply.items():
        # Match parameter names followed by colon (YAML key pattern)
        pattern = rf"^(\s*){re.escape(old_param)}(\s*):(.*)$"
        replacement = rf"\1{new_param}\2:\3"

        # Find all matches before replacing
        matches = re.findall(pattern, migrated_content, re.MULTILINE)
        if matches:
            migrated_content = re.sub(
                pattern, replacement, migrated_content, flags=re.MULTILINE
            )
            context_info = (
                f" (context: {model_type})"
                if model_type
                and old_param in CONTEXT_SPECIFIC_MAPPINGS.get(model_type, {})
                else ""
            )
            changes.append(f"{old_param} -> {new_param}{context_info}")

    return migrated_content, changes


def migrate_config_file(
    input_path: Path,
    output_path: Path | None = None,
    in_place: bool = False,
    create_backup: bool = True,
    dry_run: bool = False,
) -> bool:
    """
    Migrate a YAML configuration file.

    Args:
        input_path: Path to input configuration file
        output_path: Path for output (defaults to input_path if in_place=True)
        in_place: Whether to modify the file in place
        create_backup: Whether to create a backup file (only for in-place)
        dry_run: If True, only show what would change without modifying files

    Returns:
        True if migration was successful (or would be successful in dry-run)
    """
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return False

    if in_place:
        output_path = input_path
    elif output_path is None:
        output_path = input_path.with_stem(f"{input_path.stem}_migrated")

    try:
        # Read original file
        print(f"Reading configuration: {input_path}")
        with open(input_path, encoding="utf-8") as f:
            original_content = f.read()

        # Parse YAML to detect model type
        try:
            config_data = yaml.safe_load(original_content)
            model_type = detect_model_type_from_config(config_data)
            if model_type:
                print(f"Detected model type: {model_type}")
            else:
                print("Model type not detected, using global mappings only")
        except yaml.YAMLError as e:
            print(f"Warning: Could not parse YAML for model detection: {e}")
            print("Proceeding with global mappings only")
            model_type = None

        # Migrate content
        migrated_content, changes = migrate_yaml_content(original_content, model_type)

        if not changes:
            print("No parameter names need migration")
            return True

        print(f"Found {len(changes)} parameters to migrate:")
        for change in changes:
            print(f"  - {change}")

        if dry_run:
            print(f"[DRY RUN] Would save migrated configuration to: {output_path}")
            print("\n--- Migrated content preview ---")
            # Show a few lines around each change
            lines = migrated_content.split("\n")
            for i, line in enumerate(lines):
                for change in changes:
                    new_param = change.split(" -> ")[1].split(" (")[
                        0
                    ]  # Extract new parameter name
                    if new_param + ":" in line:
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        for j in range(start, end):
                            prefix = ">>> " if j == i else "    "
                            print(f"{prefix}{j + 1:3}: {lines[j]}")
                        print()
                        break
            return True

        # Create backup if requested and doing in-place modification
        if create_backup and in_place:
            backup_path = input_path.with_suffix(f"{input_path.suffix}.backup")
            print(f"Creating backup: {backup_path}")
            shutil.copy2(input_path, backup_path)

        # Write migrated content
        print(f"Saving migrated configuration: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(migrated_content)

        # Validate the migrated YAML
        try:
            with open(output_path, encoding="utf-8") as f:
                yaml.safe_load(f.read())
            print("Migrated YAML is valid")
        except yaml.YAMLError as e:
            print(f"Warning: Migrated YAML may have syntax issues: {e}")

        print("Migration completed successfully!")
    except Exception as e:
        print(f"Error migrating configuration: {e}")
        return False
    else:
        return True


def migrate_multiple_files(
    file_patterns: list[str],
    in_place: bool = False,
    create_backup: bool = True,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate multiple configuration files.

    Returns:
        Tuple of (successful_count, total_count)
    """
    all_files = []
    for pattern in file_patterns:
        path = Path(pattern)
        if path.is_file():
            all_files.append(path)
        else:
            # Try glob pattern
            all_files.extend(Path(".").glob(pattern))

    if not all_files:
        print("No files found matching the patterns")
        return 0, 0

    print(f"Processing {len(all_files)} configuration files...")
    successful = 0

    for file_path in all_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {file_path}")
        print("=" * 60)

        if migrate_config_file(
            input_path=file_path,
            in_place=in_place,
            create_backup=create_backup,
            dry_run=dry_run,
        ):
            successful += 1
        else:
            print(f"Failed to migrate: {file_path}")

    return successful, len(all_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate YAML configuration files to use new hyperparameter names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single config file
  python migrate_config_files.py config.yml

  # Migrate multiple files
  python migrate_config_files.py *.yml --batch

  # In-place modification
  python migrate_config_files.py config.yml --in-place

  # Dry run to see changes
  python migrate_config_files.py config.yml --dry-run

  # Specify output location
  python migrate_config_files.py config.yml --output migrated_config.yml
        """,
    )

    parser.add_argument(
        "input_files", nargs="+", help="Input configuration files or patterns"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (only for single file, ignored with --batch)",
    )

    parser.add_argument(
        "--in-place", "-i", action="store_true", help="Modify files in place"
    )

    parser.add_argument(
        "--batch", "-b", action="store_true", help="Batch process multiple files"
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files (only applies to in-place mode)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )

    args = parser.parse_args()

    if args.batch or len(args.input_files) > 1:
        if args.output:
            print("Warning: --output ignored when processing multiple files")

        successful, total = migrate_multiple_files(
            file_patterns=args.input_files,
            in_place=args.in_place,
            create_backup=not args.no_backup,
            dry_run=args.dry_run,
        )

        print(f"\n{'=' * 60}")
        print(f"Migration Summary: {successful}/{total} files processed successfully")
        sys.exit(0 if successful == total else 1)

    else:
        # Single file processing
        input_path = Path(args.input_files[0])
        success = migrate_config_file(
            input_path=input_path,
            output_path=args.output,
            in_place=args.in_place,
            create_backup=not args.no_backup,
            dry_run=args.dry_run,
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
