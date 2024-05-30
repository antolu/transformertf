"""
Utilities for implementing hyperparameter tuning with Ray Tune.

Examples
--------
The following example demonstrates how to use the `tune` module to implement
hyperparameter tuning with Ray Tune.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import typing

import lightning as L
import ray
import ray.train
import ray.train.lightning
import ray.tune
import ray.tune.integration.pytorch_lightning
import ray.tune.schedulers
import ray.tune.search
import ray.tune.search.hyperopt
import ray.tune.search.sample
import torch
import yaml

from ..data import DataModuleBase
from ..main import LightningCLI
from ..models import LightningModuleBase

__all__ = [
    "ASHATuneConfig",
    "PBTTuneConfig",
    "TuneConfig",
    "TuneReportCallback",
    "Tuneable",
    "tune",
]


@dataclasses.dataclass
class TuneConfig:
    """
    Base configuration for hyperparameter tuning.
    """

    cli_config_path: str
    """ Path to a LightningCLI config file for the parameter to be tuned. Must be a YAML file. """
    grid: dict[str, ray.tune.search.sample.Domain]
    """ Grid of hyperparameters to search over. Keys are hyperparameter names and values are Ray Tune search domains."""
    param2key: dict[str, str]
    """ Mapping from hyperparameter names in :attr:`grid` to the keys in the LightningCLI config. """
    metrics: list[str]
    """ List of metrics to report to Ray Tune. :attr:`monitor` must be present in this list. """
    monitor: str
    """ Metric to base model performance on during trials. """
    log_dir: str
    """ Directory to store logs and trials. """
    run_name: str
    """ Name of the hyperparameter tuning run. Logs will be stored in a subdirectory with this name. """
    num_epochs_per_trial: int
    """ Number of epochs to train the model for each trial at most. """
    num_samples: int
    """ Number of trials to run. """
    stop_condition: typing.Callable[[str, dict[str, float]], bool] | None = None
    """ Function to determine when to stop a trial.
    If not provided, the trial will run for :attr:`num_epochs_per_trial` epochs. or until the scheduler stops it. """

    def __post_init__(self) -> None:
        if self.monitor not in self.metrics:
            msg = (
                f"Monitor metric {self.monitor} not found in metrics. "
                "It must be present in metrics for the tune report callback to register it."
            )
            raise ValueError(msg)


@dataclasses.dataclass
class ASHATuneConfig(TuneConfig):
    patience: int = 10
    """ Number of epochs to wait for improvement before stopping a trial. """
    reduction_factor: int = 2
    """ Factor to reduce the number of epochs by when stopping a trial. """


@dataclasses.dataclass
class PBTTuneConfig(TuneConfig):
    perturbation_interval: int = 10
    """ Number of epochs to wait before perturbing the hyperparameters. """
    time_attr: str = "training_iteration"
    """ Attribute to use for for perturbation interval. """


class TuneReportCallback(
    ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback,
    L.pytorch.callbacks.Callback,
): ...


class Tuneable(ray.tune.Trainable):
    model: LightningModuleBase
    datamodule: DataModuleBase
    trainer: L.Trainer

    CHECKPOINT_FILE = "checkpoint"

    def __init__(
        self,
        tune_config: TuneConfig,
        cli_config: dict[str, typing.Any],
    ) -> None:
        super().__init__()
        self.tune_config = tune_config
        self.cli_config = cli_config

        # These will be used if the Trainable is resumed from a checkpoint
        self.checkpoint_path: str | None = None

    def setup(self, config: dict[str, typing.Any]) -> None:
        torch.set_float32_matmul_precision("high")

        new_config = apply_config(config, self.cli_config, self.tune_config.param2key)

        if "callbacks" not in new_config["trainer"]:
            new_config["trainer"]["callbacks"] = []
        tune_callback = make_tune_callback(self.tune_config.metrics)
        new_config["trainer"]["callbacks"].append(tune_callback)

        cli = LightningCLI(args=new_config, run=False)

        self.trainer = cli.trainer
        self.model = cli.model
        self.datamodule = cli.datamodule

    def step(self) -> dict[str, float]:
        # trainer = ray.train.lightning.prepare_trainer(trainer)
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.checkpoint_path)
        return {}

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(checkpoint_dir, self.CHECKPOINT_FILE)
        self.trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path)

        self.model.load_state_dict(state_dict["state_dict"])
        self.datamodule = type(self.datamodule).load_from_checkpoint(checkpoint_path)

        self.checkpoint_path = checkpoint_path


def read_from_dot_key(config: dict[str, typing.Any], key: str) -> typing.Any:
    """
    Read a value from a dot-separated key.

    Parameters
    ----------
    config : dict
        The config to read from.
    key : str
        Key to read from. E.g. "trainer.max_epochs". will return the value at
        config["trainer"]["max_epochs"].

    Returns
    -------
    Any
        The value at the key.
    """
    key_split = key.split(".")
    value = config
    for k in key_split:
        value = value[k]

    return value


def apply_key(
    config: dict[str, typing.Any], key: list[str] | str, value: typing.Any
) -> None:
    """
    Recursively apply the key to the config. If the key is a string, it will be
    split by ".", otherwise it is assumed to be a list of strings.

    Parameters
    ----------
    config : dict[str, typing.Any]
        The config to apply the key to. If the subdicts do not exist, they will
        be created.
    key : list or str
        The key to apply. If a string, it will be split by ".". For example,
        "trainer.max_epochs" will be split into ["trainer", "max_epochs"].
    value : typing.Any
        The value to apply. Can be any Python object.
    """
    if isinstance(key, str):
        key = key.split(".")

    if len(key) == 1:
        config[key[0]] = value
    else:
        if key[0] not in config:  # naively init with a dict
            config[key[0]] = {}
        apply_key(config[key[0]], key[1:], value)


def apply_config(
    config: dict[str, typing.Any],
    cli_config: dict[str, typing.Any],
    param2key: typing.Mapping[str, str],
) -> dict[str, typing.Any]:
    """
    Apply the hyperparameters to the config using the PARAM2KEY mapping.

    Parameters
    ----------
    config : dict[str, typing.Any]
        Hyperparameters to apply. For example, {"num_layers": 3, "n_dim_model": 350}.
    cli_config : dict
        The config to apply the hyperparameters to, which is a parsed config for
        LightningCLI. For example, {"trainer": {"max_epochs": 500}}.
    param2key : dict[str, str]
        Mapping from hyperparameter name to the key in the config. For example,
        {"num_layers": "model.init_args.num_layers"}.

    Returns
    -------
    dict
        The new config with the hyperparameters applied. This will be a deep copy
        of the original config.
    """
    new_config = copy.deepcopy(cli_config)

    for key, value in config.items():
        new_key = param2key[key]
        key_split = new_key.split(".")
        apply_key(new_config, key_split, value)

    return new_config


def make_tune_callback(metrics: list[str] | None = None) -> dict[str, typing.Any]:
    """Defines the TuneReportCallback in LightningCLI dict style."""
    metrics = metrics or []

    return {
        "class_path": "transformertf.utils.tune.TuneReportCallback",
        "init_args": {
            "on": "validation_end",
            "metrics": metrics,
            "save_checkpoints": False,
        },
    }


def load_cli_config(tune_config: TuneConfig) -> dict[str, typing.Any]:
    if not os.path.exists(tune_config.cli_config_path):
        msg = f"File not found: {tune_config.cli_config_path}"
        raise FileNotFoundError(msg)

    if not tune_config.cli_config_path.endswith(
        ".yaml"
    ) or not tune_config.cli_config_path.endswith(".yml"):
        msg = f"File must be a YAML file: {tune_config.cli_config_path}"
        raise ValueError(msg)

    with open(tune_config.cli_config_path, encoding="utf-8") as f:
        cli_config = yaml.full_load(f)

    cli_config.pop("ckpt_path")

    apply_key(cli_config, "trainer.max_epochs", tune_config.num_epochs_per_trial)
    apply_key(
        cli_config,
        "trainer.callbacks",
        [
            {
                "class_path": "ray.train.lightning.RayTrainReportCallback",
            }
        ],
    )
    apply_key(cli_config, "trainer.enable_progress_bar", value=False)
    apply_key(
        cli_config,
        "trainer.plugins",
        [{"class_path": "ray.train.lightning.RayLightningEnvironment"}],
    )

    return cli_config


def configure_reporter(
    grid: dict[str, ray.tune.search.sample.Domain],
    metrics: list[str],
) -> ray.tune.CLIReporter:
    return ray.tune.CLIReporter(
        parameter_columns=list(grid.keys()),
        metric_columns=metrics,
    )


def configure_search_alg(
    monitor: str,
) -> ray.tune.search.Searcher:
    return ray.tune.search.hyperopt.HyperOptSearch(
        metric=monitor,
        mode="min",
    )


def configure_asha_scheduler(
    num_epochs: int,
    patience: int,
    reduction_factor: int = 2,
) -> ray.tune.schedulers.TrialScheduler:
    return ray.tune.schedulers.ASHAScheduler(
        max_t=num_epochs,
        grace_period=patience,
        reduction_factor=reduction_factor,
    )


def configure_pbt_scheduler(
    perturbation_interval: int = 10,
    time_attr: str = "training_iteration",
) -> ray.tune.schedulers.TrialScheduler:
    return ray.tune.schedulers.PopulationBasedTraining(
        time_attr=time_attr,
        perturbation_interval=perturbation_interval,
    )


def configure_tuner(
    tune_config: TuneConfig, cli_config: dict[str, typing.Any]
) -> ray.tune.Tuner:
    reporter = configure_reporter(tune_config.grid, tune_config.metrics)
    configure_search_alg(tune_config.monitor)

    if isinstance(tune_config, ASHATuneConfig):
        scheduler = configure_asha_scheduler(
            tune_config.num_epochs_per_trial,
            tune_config.patience,
            tune_config.reduction_factor,
        )
    elif isinstance(tune_config, PBTTuneConfig):
        scheduler = configure_pbt_scheduler(
            tune_config.perturbation_interval,
            tune_config.time_attr,
        )
    else:
        msg = f"Unsupported TuneConfig type: {type(tune_config)}"
        raise TypeError(msg)

    ray_tune_config = ray.tune.TuneConfig(
        scheduler=scheduler,
        num_samples=tune_config.num_samples,
        max_concurrent_trials=torch.cuda.device_count() * 2
        if torch.cuda.is_available()
        else 4,
    )

    run_config = ray.train.RunConfig(
        name=tune_config.run_name,
        progress_reporter=reporter,
        storage_path=os.fspath(tune_config.log_dir),
        local_dir=os.fspath(tune_config.log_dir),
        stop=tune_config.stop_condition,
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_score_attribute=tune_config.monitor,
            checkpoint_score_order="min",
            num_to_keep=3,
        ),
    )

    return ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(
                Tuneable, tune_config=tune_config, cli_config=cli_config
            ),
            resources={"cpu": 4, "gpu": 1},
        ),
        tune_config=ray_tune_config,
        run_config=run_config,
    )


def tune(
    tune_config: TuneConfig,
) -> ray.tune.ResultGrid:
    """
    Tune the model with the given config and return the results.

    Parameters
    ----------
    tune_config : ASHATuneConfig | PBTTuneConfig
        The configuration for the hyperparameter tuning. This should be an
        instance of either :class:`ASHATuneConfig` or :class:`PBTTuneConfig`, which
        are subclasses of :class:`TuneConfig`.

    Returns
    -------
    ray.tune.ResultGrid
        The results of the hyperparameter tuning.
    """
    cli_config = load_cli_config(tune_config)

    tuner = configure_tuner(tune_config, cli_config)

    return tuner.fit()
