"""Common utilities for migration operations."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import torch


class MigrationError(Exception):
    """Exception raised during migration operations."""


def backup_file(file_path: Path) -> Path:
    """Create a backup of a file with .bak extension."""
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)
    return backup_path


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a PyTorch Lightning checkpoint."""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        msg = f"Error loading checkpoint {checkpoint_path}: {e}"
        raise MigrationError(msg) from e


def save_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    """Save a PyTorch Lightning checkpoint."""
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception as e:
        msg = f"Error saving checkpoint {checkpoint_path}: {e}"
        raise MigrationError(msg) from e


def load_yaml_config(config_path: Path) -> str:
    """Load a YAML configuration file as a string."""
    try:
        return config_path.read_text(encoding="utf-8")
    except Exception as e:
        msg = f"Error loading YAML config {config_path}: {e}"
        raise MigrationError(msg) from e


def save_yaml_config(content: str, config_path: Path) -> None:
    """Save a YAML configuration file."""
    try:
        config_path.write_text(content, encoding="utf-8")
    except Exception as e:
        msg = f"Error saving YAML config {config_path}: {e}"
        raise MigrationError(msg) from e


def is_checkpoint_file(file_path: Path) -> bool:
    """Check if a file is a checkpoint file."""
    return file_path.suffix.lower() in {".ckpt", ".pth", ".pt"}


def is_yaml_file(file_path: Path) -> bool:
    """Check if a file is a YAML configuration file."""
    return file_path.suffix.lower() in {".yaml", ".yml"}


def find_files_recursively(directory: Path, extensions: set[str]) -> list[Path]:
    """Find all files with specified extensions in a directory recursively."""
    files: list[Path] = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    print(f"❌ {message}", file=sys.stderr)


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"i  {message}")


def confirm_operation(message: str) -> bool:
    """Ask user for confirmation."""
    try:
        response = input(f"{message} (y/N): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return False
    else:
        return response in {"y", "yes"}
