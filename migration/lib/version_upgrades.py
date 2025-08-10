"""Version upgrade migration logic for breaking changes."""

from __future__ import annotations

import collections
from pathlib import Path

from .common import (
    MigrationError,
    backup_file,
    load_checkpoint,
    print_info,
    print_success,
    print_warning,
    save_checkpoint,
)


def migrate_v0_8_upgrade(
    checkpoint_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate checkpoint from v0.7 to v0.8 compatibility.

    Changes time_format from "relative" to "relative_legacy" due to
    StandardScaler to MaxScaler default change.

    Args:
        checkpoint_path: Path to the checkpoint file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    print_info(f"Processing v0.8 upgrade for: {checkpoint_path}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Check if datamodule_hyper_parameters exists
    if "datamodule_hyper_parameters" not in checkpoint:
        print_warning(f"No datamodule_hyper_parameters found in {checkpoint_path}")
        return False

    dm_hparams = checkpoint["datamodule_hyper_parameters"]
    if not isinstance(dm_hparams, dict):
        print_warning(
            f"Invalid datamodule_hyper_parameters structure in {checkpoint_path}"
        )
        return False

    # Check if time_format needs updating
    current_time_format = dm_hparams.get("time_format")
    if current_time_format != "relative":
        print_info(
            f"No v0.8 upgrade needed (time_format={current_time_format}): {checkpoint_path}"
        )
        return False

    migration_action = "Change time_format from 'relative' to 'relative_legacy'"

    if dry_run:
        print_warning(f"Would apply v0.8 upgrade to {checkpoint_path}:")
        print(f"  - {migration_action}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(checkpoint_path)
        print_info(f"Created backup: {backup_path}")

    # Apply the change
    checkpoint["datamodule_hyper_parameters"]["time_format"] = "relative_legacy"

    # Save updated checkpoint
    save_checkpoint(checkpoint, checkpoint_path)

    print_success(f"Applied v0.8 upgrade to {checkpoint_path}")
    print(f"  - {migration_action}")
    return True


def migrate_v0_10_upgrade(
    checkpoint_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate checkpoint from v0.9.2 to v0.10 compatibility.

    Updates MaxScaler attributes from data_min_/data_max_ to min_/max_.

    Args:
        checkpoint_path: Path to the checkpoint file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    print_info(f"Processing v0.10 upgrade for: {checkpoint_path}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Check if EncoderDecoderDataModule exists
    if "EncoderDecoderDataModule" not in checkpoint:
        print_warning(f"No EncoderDecoderDataModule found in {checkpoint_path}")
        return False

    data_module = checkpoint["EncoderDecoderDataModule"]
    if not isinstance(data_module, dict):
        print_warning(
            f"Invalid EncoderDecoderDataModule structure in {checkpoint_path}"
        )
        return False

    transforms = data_module.get("transforms", {}).get("__time__")
    if not transforms:
        print_info(f"No time transforms found in {checkpoint_path}")
        return False

    # Look for MaxScaler with old attributes
    changes = []
    transforms_updated = False

    if isinstance(transforms, dict):
        for transform_name, transform_data in transforms.items():
            if (
                isinstance(transform_data, dict)
                and transform_data.get("class_name") == "MaxScaler"
            ):
                attrs = transform_data.get("attributes", {})
                if isinstance(attrs, dict):
                    # Check for old attribute names
                    if "data_min_" in attrs and "min_" not in attrs:
                        if not dry_run:
                            attrs["min_"] = attrs.pop("data_min_")
                        changes.append(f"Rename data_min_ to min_ in {transform_name}")
                        transforms_updated = True

                    if "data_max_" in attrs and "max_" not in attrs:
                        if not dry_run:
                            attrs["max_"] = attrs.pop("data_max_")
                        changes.append(f"Rename data_max_ to max_ in {transform_name}")
                        transforms_updated = True

    if not changes:
        print_info(f"No v0.10 upgrade needed for {checkpoint_path}")
        return False

    if dry_run:
        print_warning(f"Would apply v0.10 upgrade to {checkpoint_path}:")
        for change in changes:
            print(f"  - {change}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(checkpoint_path)
        print_info(f"Created backup: {backup_path}")

    # Re-sort transforms if updated
    if transforms_updated and isinstance(transforms, dict):
        transforms = dict(sorted(transforms.items()))
        checkpoint["EncoderDecoderDataModule"]["transforms"]["__time__"] = (
            collections.OrderedDict(transforms)
        )

    # Save updated checkpoint
    save_checkpoint(checkpoint, checkpoint_path)

    print_success(f"Applied v0.10 upgrade to {checkpoint_path}")
    for change in changes:
        print(f"  - {change}")
    return True


def migrate_v0_11_upgrade(
    checkpoint_path: Path, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Migrate checkpoint from v0.10 to v0.11 compatibility.

    Adds backwards compatibility for the num_future_known_covariates calculation
    change by adding the _legacy_target_in_future_covariates flag.

    Args:
        checkpoint_path: Path to the checkpoint file
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    print_info(f"Processing v0.11 upgrade for: {checkpoint_path}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Validate checkpoint structure
    if "EncoderDecoderDataModule" not in checkpoint:
        print_warning(f"No EncoderDecoderDataModule found in {checkpoint_path}")
        return False

    if "datamodule_hyper_parameters" not in checkpoint:
        print_warning(f"No datamodule_hyper_parameters found in {checkpoint_path}")
        return False

    if "hyper_parameters" not in checkpoint:
        print_warning(f"No model hyper_parameters found in {checkpoint_path}")
        return False

    dm_hparams = checkpoint["datamodule_hyper_parameters"]
    model_hparams = checkpoint["hyper_parameters"]

    if not isinstance(dm_hparams, dict) or not isinstance(model_hparams, dict):
        print_warning(f"Invalid hyperparameters structure in {checkpoint_path}")
        return False

    # Check if already has the legacy flag
    if dm_hparams.get("_legacy_target_in_future_covariates"):
        print_info(f"Checkpoint already has legacy target flag: {checkpoint_path}")
        return False

    changes = []

    # Add the legacy flag
    migration_action = "Add _legacy_target_in_future_covariates=True flag"
    if not dry_run:
        dm_hparams["_legacy_target_in_future_covariates"] = True
    changes.append(migration_action)

    # Adjust num_future_features if present
    if (
        "num_future_features" in model_hparams
        and model_hparams["num_future_features"] > 0
    ):
        adjustment_action = f"Decrease num_future_features by 1 ({model_hparams['num_future_features']} -> {model_hparams['num_future_features'] - 1})"
        if not dry_run:
            model_hparams["num_future_features"] -= 1
        changes.append(adjustment_action)

    if dry_run:
        print_warning(f"Would apply v0.11 upgrade to {checkpoint_path}:")
        for change in changes:
            print(f"  - {change}")
        return True

    # Create backup if requested
    if backup:
        backup_path = backup_file(checkpoint_path)
        print_info(f"Created backup: {backup_path}")

    # Save updated checkpoint
    save_checkpoint(checkpoint, checkpoint_path)

    print_success(f"Applied v0.11 upgrade to {checkpoint_path}")
    for change in changes:
        print(f"  - {change}")
    return True


def migrate_version_upgrade(
    checkpoint_path: Path, version: str, *, dry_run: bool = False, backup: bool = False
) -> bool:
    """
    Apply version-specific upgrade migration to a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        version: Version to upgrade to (v0.8, v0.10, v0.11)
        dry_run: If True, only show what would be changed
        backup: If True, create a backup before modifying

    Returns:
        True if changes were made or would be made, False otherwise
    """
    version = version.lower().replace("_", ".")

    if version == "v0.8":
        return migrate_v0_8_upgrade(checkpoint_path, dry_run=dry_run, backup=backup)
    if version == "v0.10":
        return migrate_v0_10_upgrade(checkpoint_path, dry_run=dry_run, backup=backup)
    if version == "v0.11":
        return migrate_v0_11_upgrade(checkpoint_path, dry_run=dry_run, backup=backup)
    msg = (
        f"Unsupported version upgrade: {version}. "
        "Supported versions: v0.8, v0.10, v0.11"
    )
    raise MigrationError(msg)
