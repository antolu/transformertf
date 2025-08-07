"""
This script patches transformertf checkpoints moving from v0.7 to v0.8.
v0.8 changes default scaler for relative time transform from StandardScaler to MaxScaler,
which prevents loading v0.7 checkpoints with StandardScaler parameters.

v0.8 adds a time_format relative_legacy in the hparams, and this script patches the v0.7 checkpoints to change
the time_format from relative to relative_legacy if the time_format is relative.
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
        help="Path to the output patched checkpoint. Default will be the input checkpoint with '_ttf_v0.8' appended.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if "EncoderDecoderDataModule" not in checkpoint:
        msg = f"Checkpoint {args.checkpoint} does not contain EncoderDecoderDataModule, will not patch."
        print(msg, file=sys.stderr)
        return 1

    if "datamodule_hyper_parameters" not in checkpoint:
        msg = f"Checkpoint {args.checkpoint} does not contain datamodule_hyper_parameters, will not patch."
        print(msg, file=sys.stderr)
        return 1

    if "time_format" not in checkpoint["datamodule_hyper_parameters"]:
        msg = f"Checkpoint {args.checkpoint} does not contain time_format, probably using transformertf < 0.4, will not patch."
        print(msg, file=sys.stderr)
        return 1

    if checkpoint["datamodule_hyper_parameters"]["time_format"] != "relative":
        msg = f"Checkpoint {args.checkpoint} time_format is not 'relative', will not patch."
        print(msg, file=sys.stderr)
        return 1

    checkpoint["datamodule_hyper_parameters"]["time_format"] = "relative_legacy"

    output = args.output or args.checkpoint.with_name(
        args.checkpoint.stem + "_ttf_v0.8" + args.checkpoint.suffix
    )

    print(f"Saving patched checkpoint to {output}")

    torch.save(checkpoint, output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
