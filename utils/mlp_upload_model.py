#!/usr/bin/env python
"""
This script is used to upload a model checkpoint to the MLP server.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import tempfile

import mlp_client
from transformertf.main import setup_logger

from sps_mlp_hystcomp import PETEPredictor, TFTPredictor

log = logging.getLogger(__name__)


def configure_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser()

    parser.add_argument(
        "model",
        choices=["PETE", "TFT"],
        help="The model to upload checkpoint for",
    )
    parser.add_argument(
        "checkpoint", type=pathlib.Path, help="The checkpoint file to upload."
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="The name of the model to upload the checkpoint",
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        default=None,
        type=str,
        help="The version of the model to upload the checkpoint for.",
    )

    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = configure_parser()
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    setup_logger(1)
    parser = configure_parser()
    args = parser.parse_args(argv or sys.argv[1:])

    with tempfile.TemporaryDirectory() as tmp:
        client = mlp_client.Client(
            profile=mlp_client.Profile.PRO,
            temp_directory=tmp.name,  # type: ignore[attr-defined]
        )

    if args.model == "PETE":
        predictor = PETEPredictor.load_from_checkpoint(args.checkpoint)
    elif args.model == "TFT":
        predictor = TFTPredictor.load_from_checkpoint(args.checkpoint)
    else:
        msg = f"Invalid model: {args.model}"
        raise ValueError(msg)

    version = client.publish_model_parameters_version(
        predictor,
        name=args.name,
        version=args.version or mlp_client.VersionFlag.AUTO,
        verbose=True,
    )

    log.info(f"Model uploaded successfully. Version: {version}")


if __name__ == "__main__":
    main()
