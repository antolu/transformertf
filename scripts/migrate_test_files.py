#!/usr/bin/env python3
"""
Migration script for updating hyperparameter names in test files.

This script updates Python test files to use the standardized hyperparameter
naming convention:
- n_dim_model/hidden_size/hidden_dim -> d_model
- n_heads/num_attention_heads -> num_heads
- n_layers/num_lstm_layers -> num_layers

Usage:
    python scripts/migrate_test_files.py tests/
    python scripts/migrate_test_files.py tests/ --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

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
        "hidden_size": "d_model",
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


def detect_model_type_from_imports(content: str) -> str | None:
    """Detect model type from import statements."""
    # Look for model imports
    patterns = [
        r"from transformertf\.models\.(\w+)",
        r"import.*(\w+LSTM\w*)",
        r"import.*(\w+Transformer\w*)",
        r"import.*(TSMixer|PhyTSMixer)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            model_name = match.group(1)
            # Map module names to model classes
            if "attention_lstm" in model_name.lower():
                return "AttentionLSTM"
            if "temporal_conv_transformer" in model_name.lower():
                return "TemporalConvTransformer"
            if "lstm" in model_name.lower():
                return "LSTM"
            if "tsmixer" in model_name.lower():
                return "PhyTSMixer"

    return None


def migrate_python_content(
    content: str, model_type: str | None = None
) -> tuple[str, list[str]]:
    """
    Migrate Python content by replacing parameter names.

    Args:
        content: Raw Python content as string
        model_type: Detected model type for context-specific replacements

    Returns:
        Tuple of (migrated_content, list_of_changes)
    """
    migrated_content = content
    changes = []

    # Apply global parameter mappings
    for old_param, new_param in PARAMETER_MAPPINGS.items():
        # Match parameter in function calls: old_param=value
        pattern = rf"(\W){old_param}(\s*=)"
        replacement = rf"\1{new_param}\2"

        if re.search(pattern, migrated_content):
            migrated_content = re.sub(pattern, replacement, migrated_content)
            changes.append(f"{old_param} -> {new_param}")

    # Apply model-specific mappings
    if model_type and model_type in MODEL_CONTEXT_MAPPINGS:
        specific_mappings = MODEL_CONTEXT_MAPPINGS[model_type]
        for old_param, new_param in specific_mappings.items():
            pattern = rf"(\W){old_param}(\s*=)"
            replacement = rf"\1{new_param}\2"

            if re.search(pattern, migrated_content):
                migrated_content = re.sub(pattern, replacement, migrated_content)
                changes.append(f"{old_param} -> {new_param} (model-specific)")

    # Also update attribute access patterns: model.old_param or hparams.old_param
    all_mappings = {**PARAMETER_MAPPINGS}
    if model_type and model_type in MODEL_CONTEXT_MAPPINGS:
        all_mappings.update(MODEL_CONTEXT_MAPPINGS[model_type])

    for old_param, new_param in all_mappings.items():
        # Match attribute access: .old_param
        pattern = rf"(\.)({old_param})(\W)"
        replacement = rf"\1{new_param}\3"

        if re.search(pattern, migrated_content):
            migrated_content = re.sub(pattern, replacement, migrated_content)
            if f"{old_param} -> {new_param}" not in changes:
                changes.append(f"{old_param} -> {new_param} (attribute)")

    return migrated_content, changes


def migrate_test_file(
    test_path: pathlib.Path, dry_run: bool = False
) -> tuple[str, list[str]]:
    """Migrate a single test file."""
    print(f"Processing: {test_path}")

    # Read original content
    with open(test_path, encoding="utf-8") as f:
        original_content = f.read()

    # Detect model type from imports
    model_type = detect_model_type_from_imports(original_content)
    if model_type:
        print(f"  Detected model type: {model_type}")

    # Migrate content
    migrated_content, changes = migrate_python_content(original_content, model_type)

    # Display changes
    if changes:
        print("  Changes made:")
        for change in changes:
            print(f"    - {change}")
    else:
        print("  No changes needed")

    return migrated_content, changes


def find_test_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """Find all Python test files in directory."""
    test_files: list[pathlib.Path] = []
    for pattern in ["test_*.py", "*_test.py", "conftest.py"]:
        test_files.extend(directory.rglob(pattern))
    return sorted(test_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "test_directory",
        type=pathlib.Path,
        help="Directory containing test files to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
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

    args = parser.parse_args()

    if not args.test_directory.exists():
        print(f"Error: Directory {args.test_directory} does not exist", file=sys.stderr)
        return 1

    if not args.test_directory.is_dir():
        print(f"Error: {args.test_directory} is not a directory", file=sys.stderr)
        return 1

    # Find all test files
    test_files = find_test_files(args.test_directory)
    print(f"Found {len(test_files)} test files")

    successful_migrations = 0
    total_files = len(test_files)

    for test_path in test_files:
        try:
            # Migrate the file
            migrated_content, changes = migrate_test_file(
                test_path, dry_run=args.dry_run
            )

            if not changes:
                print(f"  Skipping {test_path.name} - no changes needed")
                successful_migrations += 1
                continue

            if args.dry_run:
                print(f"  Dry run: would modify {test_path}")
                continue

            # Create backup if requested
            if args.backup:
                backup_path = test_path.with_name(
                    f"{test_path.stem}_backup{test_path.suffix}"
                )
                print(f"  Creating backup: {backup_path}")
                backup_path.write_text(
                    test_path.read_text(encoding="utf-8"), encoding="utf-8"
                )

            # Write migrated content
            print(f"  Writing migrated test: {test_path}")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(migrated_content)

            successful_migrations += 1

        except Exception as e:
            print(f"Error migrating {test_path}: {e}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDry run completed - {total_files} files analyzed")
    else:
        print(
            f"\nMigration completed: {successful_migrations}/{total_files} files migrated successfully"
        )

    return 0 if successful_migrations == total_files else 1


if __name__ == "__main__":
    sys.exit(main())
