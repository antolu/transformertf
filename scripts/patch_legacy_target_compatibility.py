"""
This script patches transformertf checkpoints to add backwards compatibility
for the num_future_known_covariates calculation change.

Old checkpoints expected the target to be included in future covariates count (+1).
This script adds the _legacy_target_in_future_covariates flag to maintain compatibility.

Usage:
    python patch_legacy_target_compatibility.py checkpoint.ckpt
    python patch_legacy_target_compatibility.py checkpoint.ckpt --output checkpoint_patched.ckpt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=pathlib.Path, help="Path to the checkpoint to patch."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Path to the output patched checkpoint. Default will be the input checkpoint with '_legacy_compat' appended.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force patching even if the checkpoint appears to already have the flag.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint {args.checkpoint}: {e}", file=sys.stderr)
        return 1

    # Validate checkpoint structure
    if "EncoderDecoderDataModule" not in checkpoint:
        print(
            f"Checkpoint {args.checkpoint} does not contain EncoderDecoderDataModule, will not patch.",
            file=sys.stderr,
        )
        return 1

    if "datamodule_hyper_parameters" not in checkpoint:
        print(
            f"Checkpoint {args.checkpoint} does not contain datamodule_hyper_parameters, will not patch.",
            file=sys.stderr,
        )
        return 1

    # Check if already patched
    if (
        "_legacy_target_in_future_covariates"
        in checkpoint["datamodule_hyper_parameters"]
    ):
        if not args.force:
            current_value = checkpoint["datamodule_hyper_parameters"][
                "_legacy_target_in_future_covariates"
            ]
            print(
                f"Checkpoint {args.checkpoint} already has _legacy_target_in_future_covariates={current_value}, will not patch.",
                file=sys.stderr,
            )
            print("Use --force to override this check.", file=sys.stderr)
            return 1
        print(
            "--force specified, updating existing _legacy_target_in_future_covariates flag."
        )

    # Add the legacy compatibility flag
    print(
        "Adding _legacy_target_in_future_covariates=True to datamodule hyperparameters..."
    )
    checkpoint["datamodule_hyper_parameters"]["_legacy_target_in_future_covariates"] = (
        True
    )
    print(
        "Decreasing number of future known covariates by 1 to maintain compatibility."
    )
    checkpoint["hyper_parameters"]["num_future_features"] -= 1

    # Determine output path
    output = args.output or args.checkpoint.with_name(
        args.checkpoint.stem + "_legacy_compat" + args.checkpoint.suffix
    )

    # Save patched checkpoint
    print(f"Saving patched checkpoint to {output}")
    try:
        torch.save(checkpoint, output)
    except Exception as e:
        print(f"Error saving checkpoint {output}: {e}", file=sys.stderr)
        return 1

    print("✅ Checkpoint successfully patched for legacy target compatibility!")
    print("The patched checkpoint can now be loaded with EncoderDecoderDataModule")
    print("and will maintain the old num_future_known_covariates calculation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
