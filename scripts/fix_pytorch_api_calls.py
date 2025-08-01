#!/usr/bin/env python3
"""
Script to fix PyTorch API calls that were incorrectly changed.

PyTorch's LSTM and GRU expect `hidden_size=`, not `d_model=`.
"""

from __future__ import annotations

import pathlib
import re
import sys


def fix_pytorch_api_calls(file_path: pathlib.Path) -> tuple[str, list[str]]:
    """Fix PyTorch API calls in a single file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    changes = []

    # Fix torch.nn.LSTM calls that incorrectly use d_model= instead of hidden_size=
    lstm_pattern = r"(torch\.nn\.LSTM\([^)]*)\bd_model=([^,)]+)"
    if re.search(lstm_pattern, content):
        content = re.sub(lstm_pattern, r"\1hidden_size=\2", content)
        changes.append("torch.nn.LSTM: d_model= -> hidden_size=")

    # Fix torch.nn.GRU calls that incorrectly use d_model= instead of hidden_size=
    gru_pattern = r"(torch\.nn\.GRU\([^)]*)\bd_model=([^,)]+)"
    if re.search(gru_pattern, content):
        content = re.sub(gru_pattern, r"\1hidden_size=\2", content)
        changes.append("torch.nn.GRU: d_model= -> hidden_size=")

    return content, changes


def main() -> int:
    # Process the transformertf source directory
    source_dir = pathlib.Path("transformertf")

    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist", file=sys.stderr)
        return 1

    python_files = list(source_dir.rglob("*.py"))
    print(f"Found {len(python_files)} Python files")

    successful_fixes = 0

    for file_path in python_files:
        try:
            content, changes = fix_pytorch_api_calls(file_path)

            if changes:
                print(f"Processing: {file_path}")
                print("  Changes made:")
                for change in changes:
                    print(f"    - {change}")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  ✓ Updated {file_path}")
                successful_fixes += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    print(f"\nProcessing completed: {successful_fixes} files had PyTorch API fixes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
