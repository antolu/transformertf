"""Unified CLI for TransformerTF migration operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    MigrationError,
    confirm_operation,
    find_files_recursively,
    is_checkpoint_file,
    is_yaml_file,
    print_error,
    print_info,
    print_success,
)
from .hyperparameters import migrate_file_hyperparameters
from .scaler_inheritance import migrate_file_scaler_inheritance
from .tft_alignment import migrate_file_tft_alignment
from .version_upgrades import migrate_version_upgrade


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="migrate",
        description="Unified migration tool for TransformerTF projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate hyperparameters
  python migrate.py hyperparameters config.yaml model.ckpt

  # Migrate TFT alignment with backup
  python migrate.py tft-alignment --backup config.yaml

  # Migrate scaler inheritance
  python migrate.py scaler-inheritance model.ckpt

  # Apply version upgrade
  python migrate.py v0.11-upgrade model.ckpt

  # Process entire directory with dry run
  python migrate.py hyperparameters --directory ./configs --dry-run

  # Chain operations easily (same file paths)
  python migrate.py hyperparameters model.ckpt
  python migrate.py tft-alignment model.ckpt
  python migrate.py v0.11-upgrade model.ckpt
""",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Migration command to run",
        required=True,
        metavar="COMMAND",
    )

    # Common arguments for all subcommands
    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "files",
            nargs="*",
            type=Path,
            help="Files to migrate (.ckpt, .pth, .pt, .yaml, .yml)",
        )
        subparser.add_argument(
            "--directory", "-d", type=Path, help="Directory to migrate recursively"
        )
        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be changed without modifying files",
        )
        subparser.add_argument(
            "--backup", action="store_true", help="Create .bak files before modifying"
        )
        subparser.add_argument(
            "--force", action="store_true", help="Skip confirmation prompts"
        )

    # Hyperparameters subcommand
    hyper_parser = subparsers.add_parser(
        "hyperparameters",
        help="Migrate hyperparameter names (v0.12 standardization)",
        description="Migrate hyperparameter names to standardized conventions",
    )
    add_common_args(hyper_parser)

    # TFT alignment subcommand
    tft_parser = subparsers.add_parser(
        "tft-alignment",
        help="Migrate TFT encoder alignment (v0.13 compatibility)",
        description="Add explicit encoder_alignment='right' for TFT-family models",
    )
    add_common_args(tft_parser)

    # Scaler inheritance subcommand
    scaler_parser = subparsers.add_parser(
        "scaler-inheritance",
        help="Migrate scaler inheritance structure (MaxScaler/MinMaxScaler refactor)",
        description="Update models with old scaler inheritance to new structure",
    )
    add_common_args(scaler_parser)

    # Version upgrade subcommands
    for version, desc in [
        ("v0.8-upgrade", "v0.7 to v0.8 compatibility (time_format fix)"),
        ("v0.10-upgrade", "v0.9.2 to v0.10 compatibility (MaxScaler attributes)"),
        ("v0.11-upgrade", "v0.10 to v0.11 compatibility (legacy target flag)"),
    ]:
        version_parser = subparsers.add_parser(
            version, help=desc, description=f"Apply {desc}"
        )
        add_common_args(version_parser)

    return parser


def get_files_to_process(args: argparse.Namespace) -> list[Path]:
    """Get list of files to process based on arguments."""
    files = []

    if args.directory:
        if not args.directory.exists():
            msg = f"Directory does not exist: {args.directory}"
            raise MigrationError(msg)
        if not args.directory.is_dir():
            msg = f"Path is not a directory: {args.directory}"
            raise MigrationError(msg)

        # Find all supported files in directory
        extensions = {".ckpt", ".pth", ".pt", ".yaml", ".yml"}
        files.extend(find_files_recursively(args.directory, extensions))

        if not files:
            print_info(f"No supported files found in {args.directory}")
            return []

    elif args.files:
        for file_path in args.files:
            if not file_path.exists():
                msg = f"File does not exist: {file_path}"
                raise MigrationError(msg)
            if not file_path.is_file():
                msg = f"Path is not a file: {file_path}"
                raise MigrationError(msg)
            if not (is_checkpoint_file(file_path) or is_yaml_file(file_path)):
                msg = (
                    f"Unsupported file type: {file_path.suffix}. "
                    "Supported: .ckpt, .pth, .pt, .yaml, .yml"
                )
                raise MigrationError(msg)
            files.append(file_path)
    else:
        msg = "Must specify either files or --directory"
        raise MigrationError(msg)

    return files


def run_hyperparameters_migration(args: argparse.Namespace) -> int:
    """Run hyperparameter name migration."""
    files = get_files_to_process(args)
    if not files:
        return 0

    print_info(f"Processing {len(files)} files for hyperparameter migration")

    if (
        not args.dry_run
        and not args.force
        and not confirm_operation(f"Migrate hyperparameters in {len(files)} files?")
    ):
        return 1

    success_count = 0
    total_count = len(files)

    for file_path in files:
        try:
            if migrate_file_hyperparameters(
                file_path, dry_run=args.dry_run, backup=args.backup
            ):
                success_count += 1
        except MigrationError as e:
            print_error(f"Migration failed for {file_path}: {e}")
            return 1
        except Exception as e:
            print_error(f"Unexpected error processing {file_path}: {e}")
            return 1

    if args.dry_run:
        print_success(
            f"Dry run completed. Would migrate {success_count}/{total_count} files"
        )
    else:
        print_success(
            f"Migration completed. Processed {success_count}/{total_count} files"
        )

    return 0


def run_tft_alignment_migration(args: argparse.Namespace) -> int:
    """Run TFT encoder alignment migration."""
    files = get_files_to_process(args)
    if not files:
        return 0

    print_info(f"Processing {len(files)} files for TFT alignment migration")

    if (
        not args.dry_run
        and not args.force
        and not confirm_operation(f"Migrate TFT alignment in {len(files)} files?")
    ):
        return 1

    success_count = 0
    total_count = len(files)

    for file_path in files:
        try:
            if migrate_file_tft_alignment(
                file_path, dry_run=args.dry_run, backup=args.backup
            ):
                success_count += 1
        except MigrationError as e:
            print_error(f"Migration failed for {file_path}: {e}")
            return 1
        except Exception as e:
            print_error(f"Unexpected error processing {file_path}: {e}")
            return 1

    if args.dry_run:
        print_success(
            f"Dry run completed. Would migrate {success_count}/{total_count} files"
        )
    else:
        print_success(
            f"Migration completed. Processed {success_count}/{total_count} files"
        )

    return 0


def run_scaler_inheritance_migration(args: argparse.Namespace) -> int:
    """Run scaler inheritance migration."""
    files = get_files_to_process(args)
    if not files:
        return 0

    # Only process checkpoint files for scaler inheritance migration
    checkpoint_files = [f for f in files if is_checkpoint_file(f)]
    if not checkpoint_files:
        print_info("No checkpoint files found for scaler inheritance migration")
        return 0

    print_info(
        f"Processing {len(checkpoint_files)} checkpoints for scaler inheritance migration"
    )

    if (
        not args.dry_run
        and not args.force
        and not confirm_operation(
            f"Migrate scaler inheritance in {len(checkpoint_files)} checkpoints?"
        )
    ):
        return 1

    success_count = 0
    total_count = len(checkpoint_files)

    for checkpoint_path in checkpoint_files:
        try:
            if migrate_file_scaler_inheritance(
                checkpoint_path, dry_run=args.dry_run, backup=args.backup
            ):
                success_count += 1
        except MigrationError as e:
            print_error(f"Migration failed for {checkpoint_path}: {e}")
            return 1
        except Exception as e:
            print_error(f"Unexpected error processing {checkpoint_path}: {e}")
            return 1

    if args.dry_run:
        print_success(
            f"Dry run completed. Would migrate {success_count}/{total_count} checkpoints"
        )
    else:
        print_success(
            f"Migration completed. Processed {success_count}/{total_count} checkpoints"
        )

    return 0


def run_version_upgrade(args: argparse.Namespace) -> int:
    """Run version-specific upgrade migration."""
    files = get_files_to_process(args)
    if not files:
        return 0

    # Only process checkpoint files for version upgrades
    checkpoint_files = [f for f in files if is_checkpoint_file(f)]
    if not checkpoint_files:
        print_info("No checkpoint files found for version upgrade")
        return 0

    # Extract version from command name
    version = args.command.split("-")[0]  # e.g., "v0.11-upgrade" -> "v0.11"

    print_info(f"Processing {len(checkpoint_files)} checkpoints for {version} upgrade")

    if (
        not args.dry_run
        and not args.force
        and not confirm_operation(
            f"Apply {version} upgrade to {len(checkpoint_files)} checkpoints?"
        )
    ):
        return 1

    success_count = 0
    total_count = len(checkpoint_files)

    for checkpoint_path in checkpoint_files:
        try:
            if migrate_version_upgrade(
                checkpoint_path, version, dry_run=args.dry_run, backup=args.backup
            ):
                success_count += 1
        except MigrationError as e:
            print_error(f"Migration failed for {checkpoint_path}: {e}")
            return 1
        except Exception as e:
            print_error(f"Unexpected error processing {checkpoint_path}: {e}")
            return 1

    if args.dry_run:
        print_success(
            f"Dry run completed. Would upgrade {success_count}/{total_count} checkpoints"
        )
    else:
        print_success(
            f"Upgrade completed. Processed {success_count}/{total_count} checkpoints"
        )

    return 0


def main() -> int:
    """Main CLI entry point."""
    try:
        parser = create_parser()
        args = parser.parse_args()

        # Route to appropriate handler
        if args.command == "hyperparameters":
            return run_hyperparameters_migration(args)
        if args.command == "tft-alignment":
            return run_tft_alignment_migration(args)
        if args.command == "scaler-inheritance":
            return run_scaler_inheritance_migration(args)
        if args.command.endswith("-upgrade"):
            return run_version_upgrade(args)
        print_error(f"Unknown command: {args.command}")
        return 1  # noqa: TRY300

    except MigrationError as e:
        print_error(f"Migration error: {e}")
        return 1
    except KeyboardInterrupt:
        print_error("Migration cancelled by user")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 1
