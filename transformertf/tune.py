from __future__ import annotations

import copy
import functools
import logging
import sys
import typing

import jsonargparse
import lightning as L
import ray.tune
import ray.tune.integration.pytorch_lightning
import ray.tune.search.sample
import yaml

from transformertf.data import (
    EncoderDataModule,  # noqa: F401
    EncoderDecoderDataModule,  # noqa: F401
    TimeSeriesDataModule,  # noqa: F401
)
from transformertf.models.bwlstm import BWLSTM1, BWLSTM2, BWLSTM3  # noqa: F401
from transformertf.models.lstm import LSTM  # noqa: F401
from transformertf.models.temporal_fusion_transformer import (  # noqa: F401
    TemporalFusionTransformer,
)
from transformertf.models.tsmixer import TSMixer  # noqa: F401

from .main import LightningCLI

log = logging.getLogger(__name__)


class TuneReportCallback(
    ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback,
    L.pytorch.callbacks.Callback,
): ...


SUPPORTED_DISTRS = (
    ray.tune.uniform,
    ray.tune.quniform,
    ray.tune.loguniform,
    ray.tune.qloguniform,
    ray.tune.randint,
    ray.tune.qrandint,
    ray.tune.choice,
    ray.tune.randn,
    ray.tune.qrandn,
    ray.tune.lograndint,
    ray.tune.qlograndint,
    ray.tune.grid_search,
)


@functools.lru_cache
def distr_to_fn(distr: str) -> typing.Callable:
    for d in SUPPORTED_DISTRS:
        if d.__name__.lower() == distr.lower():
            return d
    msg = f"Unknown distribution: {distr}"
    raise ValueError(msg)


def function_from_string(
    distr_name: str, **init_args: typing.Any
) -> ray.tune.search.sample.Domain:
    return distr_to_fn(distr_name)(**init_args)


def class_from_function(
    function: typing.Callable[[str, typing.Any], ray.tune.search.sample.Domain],
) -> typing.Callable[[str, typing.Any], ray.tune.search.sample.Domain]:
    return jsonargparse.class_from_function(
        function,
        func_return=ray.tune.search.sample.Domain,
        name=f"ray.tune.{function.__name__}",
    )


DistributionFactory = jsonargparse.class_from_function(function_from_string)


class TunerLightningCLI(LightningCLI):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if "run" in kwargs:
            log.warning(
                "TunerLightningCLI will set run to 'False' to prevent running the model."
            )
        kwargs["run"] = False
        super().__init__(*args, **kwargs)

    # def add


def objective(config: dict[str, typing.Any]) -> float:
    LightningCLI(args=config)

    return 0.0


def main(argv: list[str] | None = None) -> None:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file. Must be compliant with LightningCLI",
    )

    args = parser.parse_args(argv)

    with open(args.config, encoding="locale") as f:
        config = yaml.full_load(f)

    if "tuner" not in config:
        msg = "Configuration file must have a 'tuner' section."
        raise ValueError(msg)

    config_lightning = copy.deepcopy(config)
    config_lightning.pop("tuner")

    parser = jsonargparse.ArgumentParser()

    def add_hparam(hparam: str, default: typing.Any) -> None:
        parser.add_argument(
            f"--{hparam}",
            type=class_from_function(ray.tune.uniform),
            help=f"Hyperparameter {hparam}",
        )

    def add_init_args_recursive(d: dict[str, typing.Any], stem: str = "") -> None:
        if stem:
            stem = f"{stem}."
        if "init_args" in d:
            for k, v in d["init_args"].items():
                if isinstance(v, dict) and "init_args" in v:
                    add_init_args_recursive(v, f"{stem}{k}")
                add_hparam(f"{stem}{k}", v)

    add_init_args_recursive(config["model"], "model")

    args = parser.parse_object(config["tuner"])

    breakpoint()


if __name__ == "__main__":
    main(sys.argv[1:])
