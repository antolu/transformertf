#!/usr/bin/env python3
"""
Project-wide migration script for TransformerTF hyperparameter names.

This script finds and migrates all checkpoint and configuration files in a project
directory to use the new standardized hyperparameter naming conventions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the individual migration functions
try:
    from migrate_config_files import migrate_config_file
    from migrate_hyperparameter_names import migrate_checkpoint
except ImportError:
    print("Error: Migration scripts not found in the same directory")
    sys.exit(1)


def find_checkpoint_files(root_dir: Path, recursive: bool = True) -> list[Path]:
    """Find all checkpoint files in the directory."""
    patterns = ["*.ckpt", "*.pth"]
    files = []

    for pattern in patterns:
        if recursive:
            files.extend(root_dir.rglob(pattern))
        else:
            files.extend(root_dir.glob(pattern))

    # Filter out backup files and already migrated files
    filtered_files = []
    for file_path in files:
        if file_path.name.endswith(".backup"):
            continue
        if "migrated" in file_path.name:
            continue
        filtered_files.append(file_path)

    return sorted(filtered_files)


def find_config_files(root_dir: Path, recursive: bool = True) -> list[Path]:
    """Find all configuration files in the directory."""
    patterns = ["*.yml", "*.yaml"]
    files = []

    for pattern in patterns:
        if recursive:
            files.extend(root_dir.rglob(pattern))
        else:
            files.extend(root_dir.glob(pattern))

    # Filter out backup files and already migrated files
    filtered_files = []
    for file_path in files:
        if file_path.name.endswith(".backup"):
            continue
        if "migrated" in file_path.name:
            continue
        # Skip some common non-config YAML files
        if file_path.name in [".pre-commit-config.yaml", ".github/workflows/*.yml"]:
            continue
        filtered_files.append(file_path)

    return sorted(filtered_files)


def migrate_project_files(
    root_dir: Path,
    checkpoints_only: bool = False,
    configs_only: bool = False,
    recursive: bool = True,
    create_backup: bool = True,
    dry_run: bool = False,
) -> dict[str, tuple[int, int]]:
    """
    Migrate all relevant files in a project directory.

    Returns:
        Dictionary with migration results: {'checkpoints': (success, total), 'configs': (success, total)}
    """
    results = {"checkpoints": (0, 0), "configs": (0, 0)}

    if not configs_only:
        # Find and migrate checkpoint files
        checkpoint_files = find_checkpoint_files(root_dir, recursive)
        if checkpoint_files:
            print(f"\n{'=' * 60}")
            print(f"CHECKPOINT FILES ({len(checkpoint_files)} found)")
            print("=" * 60)

            successful_checkpoints = 0
            for ckpt_file in checkpoint_files:
                print(f"\nProcessing checkpoint: {ckpt_file.relative_to(root_dir)}")
                print("-" * 50)

                if migrate_checkpoint(
                    input_path=ckpt_file, create_backup=create_backup, dry_run=dry_run
                ):
                    successful_checkpoints += 1
                else:
                    print(f"Failed to migrate: {ckpt_file}")

            results["checkpoints"] = (successful_checkpoints, len(checkpoint_files))
        else:
            print("No checkpoint files found")

    if not checkpoints_only:
        # Find and migrate configuration files
        config_files = find_config_files(root_dir, recursive)
        if config_files:
            print(f"\n{'=' * 60}")
            print(f"CONFIGURATION FILES ({len(config_files)} found)")
            print("=" * 60)

            successful_configs = 0
            for config_file in config_files:
                print(f"\nProcessing config: {config_file.relative_to(root_dir)}")
                print("-" * 50)

                if migrate_config_file(
                    input_path=config_file,
                    in_place=True,
                    create_backup=create_backup,
                    dry_run=dry_run,
                ):
                    successful_configs += 1
                else:
                    print(f"Failed to migrate: {config_file}")

            results["configs"] = (successful_configs, len(config_files))
        else:
            print("No configuration files found")

    return results


def print_summary(results: dict[str, tuple[int, int]], dry_run: bool = False):
    """Print migration summary."""
    action = "Would be processed" if dry_run else "Processed"

    print(f"\n{'=' * 60}")
    print("MIGRATION SUMMARY")
    print("=" * 60)

    total_successful = 0
    total_files = 0

    for file_type, (successful, total) in results.items():
        if total > 0:
            percentage = (successful / total) * 100
            print(
                f"{file_type.title()}: {successful}/{total} {action.lower()} successfully ({percentage:.1f}%)"
            )
            total_successful += successful
            total_files += total

    if total_files > 0:
        overall_percentage = (total_successful / total_files) * 100
        print(
            f"\nOverall: {total_successful}/{total_files} files {action.lower()} successfully ({overall_percentage:.1f}%)"
        )
    else:
        print("No files found to migrate")

    if not dry_run:
        if total_successful == total_files and total_files > 0:
            print("\n✅ All files migrated successfully!")
        elif total_successful > 0:
            print(
                f"\n⚠️  {total_files - total_successful} files had issues during migration"
            )
        else:
            print("\n❌ No files were migrated successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate all TransformerTF files in a project directory to use new hyperparameter names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate entire project
  python migrate_project.py /path/to/project

  # Migrate current directory
  python migrate_project.py .

  # Dry run to see what would be migrated
  python migrate_project.py /path/to/project --dry-run

  # Only migrate checkpoints
  python migrate_project.py /path/to/project --checkpoints-only

  # Only migrate config files
  python migrate_project.py /path/to/project --configs-only

  # Non-recursive (current directory only)
  python migrate_project.py /path/to/project --no-recursive
        """,
    )

    parser.add_argument(
        "project_dir", type=Path, help="Path to project directory to migrate"
    )

    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Only migrate checkpoint files (.ckpt, .pth)",
    )

    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Only migrate configuration files (.yml, .yaml)",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recursively search subdirectories",
    )

    parser.add_argument(
        "--no-backup", action="store_true", help="Don't create backup files"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.checkpoints_only and args.configs_only:
        print("Error: Cannot specify both --checkpoints-only and --configs-only")
        sys.exit(1)

    if not args.project_dir.exists():
        print(f"Error: Directory {args.project_dir} does not exist")
        sys.exit(1)

    if not args.project_dir.is_dir():
        print(f"Error: {args.project_dir} is not a directory")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("TransformerTF Project Migration")
    print(f"{'=' * 60}")
    print(f"Project directory: {args.project_dir.absolute()}")
    print(f"Recursive search: {not args.no_recursive}")
    print(f"Create backups: {not args.no_backup}")
    print(f"Dry run: {args.dry_run}")

    if args.checkpoints_only:
        print("Mode: Checkpoints only")
    elif args.configs_only:
        print("Mode: Configuration files only")
    else:
        print("Mode: All files")

    # Run migration
    results = migrate_project_files(
        root_dir=args.project_dir,
        checkpoints_only=args.checkpoints_only,
        configs_only=args.configs_only,
        recursive=not args.no_recursive,
        create_backup=not args.no_backup,
        dry_run=args.dry_run,
    )

    # Print summary
    print_summary(results, args.dry_run)

    # Exit with appropriate code
    total_successful = sum(successful for successful, _ in results.values())
    total_files = sum(total for _, total in results.values())

    if total_files == 0:
        # No files found - not an error
        sys.exit(0)
    elif total_successful == total_files:
        # All files processed successfully
        sys.exit(0)
    else:
        # Some files failed
        sys.exit(1)


if __name__ == "__main__":
    main()
