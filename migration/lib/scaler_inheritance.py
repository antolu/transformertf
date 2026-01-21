"""Scaler inheritance migration for TransformerTF models.

This module handles migration of models that contain scalers with the old
inheritance structure where MinMaxScaler was the base class and MaxScaler
inherited from it. The new structure has MaxScaler as the base class and
MinMaxScaler inheriting from it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .common import MigrationError, backup_file, print_info, print_success

log = logging.getLogger(__name__)


def needs_scaler_migration(data: dict[str, Any]) -> bool:
    """
    Check if the data structure needs scaler inheritance migration.

    This is a heuristic check since the actual inheritance structure
    isn't directly visible in the serialized data. We look for patterns
    that suggest the old structure.

    Parameters
    ----------
    data : dict
        The loaded checkpoint or state dict.

    Returns
    -------
    bool
        True if migration is needed, False otherwise.
    """
    # Look for scaler-related keys in state dict
    state_dict = data.get("state_dict", data)

    scaler_keys = [
        key
        for key in state_dict
        if any(scaler_type in key for scaler_type in ["MaxScaler", "MinMaxScaler"])
    ]

    if scaler_keys:
        log.info(f"Found {len(scaler_keys)} scaler-related parameters")
        # For now, we assume all models with scalers might need migration
        # In a more sophisticated version, we could check the actual class
        # hierarchy stored in the model metadata
        return True

    # Check transforms in data modules
    return "datamodule" in data


def migrate_scaler_data_structure(data: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate the data structure to be compatible with new scaler inheritance.

    Since the parameter names and functionality remain the same,
    this is primarily about ensuring compatibility with the new
    class hierarchy.

    Parameters
    ----------
    data : dict[str, Any]
        The original data structure.

    Returns
    -------
    dict[str, Any]
        The migrated data structure.
    """
    # Create a deep copy to avoid modifying the original
    migrated: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            migrated[key] = migrate_scaler_data_structure(value)
        else:
            migrated[key] = value

    # Add migration metadata
    migrated["_scaler_migration_applied"] = True
    migrated["_migration_version"] = "1.0.0"

    log.info("Applied scaler inheritance migration")
    return migrated


def migrate_file_scaler_inheritance(
    file_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate scaler inheritance in a checkpoint or state dict file.

    Parameters
    ----------
    file_path : Path
        Path to the file to migrate (.ckpt, .pth, .pt)
    dry_run : bool
        If True, show what would be changed without modifying files
    backup : bool
        If True, create .bak backup before modifying

    Returns
    -------
    bool
        True if file was migrated (or would be migrated in dry run mode),
        False if no migration was needed.

    Raises
    ------
    MigrationError
        If migration fails or file is invalid.
    """
    if file_path.suffix not in {".ckpt", ".pth", ".pt"}:
        return False  # Skip non-checkpoint files

    try:
        print_info(f"Checking {file_path.name}")

        # Load the file
        data = torch.load(file_path, map_location="cpu")

        # Check if migration is needed
        if not needs_scaler_migration(data):
            log.info(f"No scaler migration needed for {file_path.name}")
            return False

        if dry_run:
            print_info(f"Would migrate scaler inheritance in {file_path.name}")
            return True

        # Create backup if requested
        if backup:
            backup_file(file_path)

        # Perform migration
        migrated_data = migrate_scaler_data_structure(data)

        # Save migrated file
        torch.save(migrated_data, file_path)
        print_success(f"Migrated scaler inheritance in {file_path.name}")
    except Exception as e:
        msg = f"Failed to migrate scaler inheritance in {file_path}: {e}"
        raise MigrationError(msg) from e
    else:
        return True
