#!/usr/bin/env python3
"""
Script to fix parameter usage in function calls within the codebase.

This fixes cases where we're calling functions with old parameter names like:
    InterpretableMultiHeadAttention(n_heads=..., n_dim_model=...)

Should become:
    InterpretableMultiHeadAttention(num_heads=..., d_model=...)
"""

from __future__ import annotations

import pathlib
import sys

# Mapping of old parameter names to new ones in function calls
CALL_PARAMETER_MAPPINGS = {
    "n_heads=": "num_heads=",
    "n_dim_model=": "d_model=",
    "hidden_size=": "d_model=",  # Context-specific for LSTM models
    "num_attention_heads=": "num_heads=",
    "n_layers=": "num_layers=",
}

# Context-specific mappings (only apply in certain contexts)
CONTEXT_SPECIFIC_MAPPINGS = {
    # For LSTM models, hidden_size should become d_model
    "LSTM": {
        "hidden_size=": "d_model=",
    },
    # But for other contexts, hidden_size might remain as-is
}


def fix_parameter_usage_in_file(
    file_path: pathlib.Path, dry_run: bool = False
) -> tuple[str, list[str]]:
    """Fix parameter usage in a single file."""
    print(f"Processing: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    changes = []

    # Apply standard mappings
    for old_param, new_param in CALL_PARAMETER_MAPPINGS.items():
        if old_param in content:
            content = content.replace(old_param, new_param)
            changes.append(f"{old_param} -> {new_param}")

    if changes:
        print("  Changes made:")
        for change in changes:
            print(f"    - {change}")

        if not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✓ Updated {file_path}")
    else:
        print("  No changes needed")

    return content, changes


def find_python_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """Find all Python files in directory."""
    python_files: list[pathlib.Path] = []
    for pattern in ["*.py"]:
        python_files.extend(directory.rglob(pattern))
    return sorted(python_files)


def main() -> int:
    # Process the transformertf source directory
    source_dir = pathlib.Path("transformertf")

    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist", file=sys.stderr)
        return 1

    python_files = find_python_files(source_dir)
    print(f"Found {len(python_files)} Python files")

    successful_fixes = 0
    len(python_files)

    for file_path in python_files:
        try:
            _content, changes = fix_parameter_usage_in_file(file_path)

            if changes:
                successful_fixes += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    print(f"\nProcessing completed: {successful_fixes} files had parameter usage fixes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
