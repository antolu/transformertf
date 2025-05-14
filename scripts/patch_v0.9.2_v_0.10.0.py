"""
This script patches transformertf checkpoints moving from v0.9.2 to v0.10.0.
v0.8 changes the MaxScaler to scale using min_ and max_ instead of data_min_ and data_max_.
"""

from __future__ import annotations

import argparse
import collections
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

    if "__time__" not in checkpoint["EncoderDecoderDataModule"]["transforms"]:
        msg = f"Checkpoint {args.checkpoint} does not contain __time__ transform, will not patch."
        print(msg, file=sys.stderr)
        return 1

    if (
        "transforms.1.max_"
        in checkpoint["EncoderDecoderDataModule"]["transforms"]["__time__"]
        and "transforms.1.min_"
        not in checkpoint["EncoderDecoderDataModule"]["transforms"]["__time__"]
    ):
        transforms = checkpoint["EncoderDecoderDataModule"]["transforms"]["__time__"]

        transforms["transforms.1.data_min_"] = torch.tensor([0.0])
        transforms["transforms.1.data_max_"] = transforms["transforms.1.max_"]
        transforms["transforms.1.min_"] = torch.tensor([0.0])
        transforms["transforms.1.max_"] = torch.tensor([1.0])
        transforms["transforms.1.frozen_"] = torch.tensor([False])
        transforms["transforms.1.num_features_"] = torch.tensor([1])

        # sort the keys
        transforms = dict(sorted(transforms.items()))
        checkpoint["EncoderDecoderDataModule"]["transforms"]["__time__"] = (
            collections.OrderedDict(transforms)
        )

    output = args.output or args.checkpoint.with_name(
        args.checkpoint.stem + "_ttf_v0.10.0" + args.checkpoint.suffix
    )

    print(f"Saving patched checkpoint to {output}")

    torch.save(checkpoint, output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
