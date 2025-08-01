"""
Batch migration script for updating all hyperparameter names in a TransformerTF project.

This script finds and migrates all checkpoint files (.ckpt, .pth) and configuration
files (.yml, .yaml) in a directory to use the standardized hyperparameter naming
convention.

Usage:
    python scripts/migrate_project.py /path/to/project
    python scripts/migrate_project.py . --dry-run
    python scripts/migrate_project.py /path/to/project --no-backup
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def find_files(directory: pathlib.Path, patterns: list[str]) -> list[pathlib.Path]:
    """Find files matching the given patterns in directory."""
    files: list[pathlib.Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted(files)


def run_migration_script(
    script_path: pathlib.Path, files: list[pathlib.Path], args: argparse.Namespace
) -> bool:
    """Run a migration script on a list of files."""
    if not files:
        return True

    script_args = [str(file) for file in files]

    # Add common arguments
    if args.dry_run:
        script_args.append("--dry-run")
    if not args.backup:
        script_args.append("--no-backup")
    if len(files) > 1:
        script_args.append("--batch")

    # Run the migration script
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), *script_args],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path.name}: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False
    else:
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directory", type=pathlib.Path, help="Directory to search for files to migrate"
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
    parser.add_argument(
        "--checkpoints-only", action="store_true", help="Only migrate checkpoint files"
    )
    parser.add_argument(
        "--configs-only", action="store_true", help="Only migrate configuration files"
    )

    args = parser.parse_args(argv or sys.argv[1:])

    if not args.directory.exists():
        print(f"Error: Directory {args.directory} does not exist", file=sys.stderr)
        return 1

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        return 1

    # Get script directory
    scripts_dir = pathlib.Path(__file__).parent

    print(f"Searching for files in: {args.directory}")

    success = True

    # Migrate checkpoint files
    if not args.configs_only:
        print("\n" + "=" * 50)
        print("MIGRATING CHECKPOINT FILES")
        print("=" * 50)

        checkpoint_files = find_files(args.directory, ["*.ckpt", "*.pth"])
        print(f"Found {len(checkpoint_files)} checkpoint files")

        if checkpoint_files:
            checkpoint_script = scripts_dir / "migrate_hyperparameter_names.py"
            if not checkpoint_script.exists():
                print(
                    f"Error: Migration script {checkpoint_script} not found",
                    file=sys.stderr,
                )
                success = False
            else:
                success &= run_migration_script(
                    checkpoint_script, checkpoint_files, args
                )
        else:
            print("No checkpoint files found")

    # Migrate configuration files
    if not args.checkpoints_only:
        print("\n" + "=" * 50)
        print("MIGRATING CONFIGURATION FILES")
        print("=" * 50)

        config_files = find_files(args.directory, ["*.yml", "*.yaml"])
        # Filter out backup files and migrated files
        config_files = [
            f
            for f in config_files
            if not (f.stem.endswith("_backup") or f.stem.endswith("_migrated"))
        ]
        print(f"Found {len(config_files)} configuration files")

        if config_files:
            config_script = scripts_dir / "migrate_config_files.py"
            if not config_script.exists():
                print(
                    f"Error: Migration script {config_script} not found",
                    file=sys.stderr,
                )
                success = False
            else:
                success &= run_migration_script(config_script, config_files, args)
        else:
            print("No configuration files found")

    if args.dry_run:
        print(f"\n{'=' * 50}")
        print("DRY RUN COMPLETED")
        print(f"{'=' * 50}")
        print("No files were modified. Use without --dry-run to apply changes.")
    else:
        status = "SUCCESS" if success else "COMPLETED WITH ERRORS"
        print(f"\n{'=' * 50}")
        print(f"MIGRATION {status}")
        print(f"{'=' * 50}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
