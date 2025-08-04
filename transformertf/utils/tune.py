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
import logging
import os
import pprint
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

from ..callbacks import SetOptimizerParamsCallback
from ..main import LightningCLI

log = logging.getLogger(__name__)


__all__ = [
    "ASHATuneConfig",
    "PBTTuneConfig",
    "TuneConfig",
    "TuneReportCallback",
    "tune",
    "tune_fn",
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


def tune_fn(
    config: dict[str, typing.Any],
    tune_config: TuneConfig,
    cli_config: dict[str, typing.Any],
) -> dict[str, float]:
    torch.set_float32_matmul_precision("high")

    new_config = apply_config(config, cli_config, tune_config.param2key)

    if "callbacks" not in new_config["trainer"]:
        new_config["trainer"]["callbacks"] = []
    tune_callback = make_tune_callback(tune_config.metrics)
    new_config["trainer"]["callbacks"].append(tune_callback)

    if isinstance(tune_config, PBTTuneConfig):
        set_optimizer_params_callback = make_set_optimizer_params_callback(cli_config)
        new_config["trainer"]["callbacks"].append(set_optimizer_params_callback)

    log.info(f"New trial config: {pprint.pformat(new_config)}")

    cli = LightningCLI(args=new_config, run=False)

    trainer = cli.trainer
    model = cli.model
    datamodule = cli.datamodule

    # If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
    if (
        isinstance(tune_config, PBTTuneConfig)
        and (checkpoint := ray.train.get_checkpoint()) is not None
    ):
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict["state_dict"])
            datamodule = type(datamodule).load_from_checkpoint(checkpoint_path)

        optimizer_callback = find_optimizer_callback(trainer.callbacks)
        if optimizer_callback is None:
            msg = "Optimizer parameter callback not found."
            raise ValueError(msg)

        update_optimizer_params(optimizer_callback, cli_config)
    else:
        checkpoint_path = None

    trainer.fit(model, datamodule, ckpt_path=checkpoint_path)

    return {}


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


def find_optimizer_params(
    cli_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Find the optimizer parameters in the LightningCLI config."""
    optimizer_params = {}
    if "optimizer" in cli_config and "init_args" in cli_config["optimizer"]:
        optimizer_params = cli_config["optimizer"]["init_args"]

    return optimizer_params


def find_lr_scheduler_params(
    cli_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Find the LR scheduler parameters in the LightningCLI config."""
    lr_params = {}
    if "lr_scheduler" in cli_config and "init_args" in cli_config["lr_scheduler"]:
        lr_params = cli_config["lr_scheduler"]["init_args"]

    return lr_params


def make_set_optimizer_params_callback(
    cli_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Defines the SetOptimizerParamsCallback in LightningCLI dict style."""
    return {
        "class_path": "transformertf.callbacks.SetOptimizerParamsCallback",
        "init_args": find_optimizer_params(cli_config)
        | find_lr_scheduler_params(cli_config),
    }


def update_optimizer_params(
    callback: SetOptimizerParamsCallback,
    cli_config: dict[str, typing.Any],
) -> None:
    """Update the optimizer parameters in the SetOptimizerParamsCallback."""
    optimizer_params = find_optimizer_params(cli_config)
    lr_params = find_lr_scheduler_params(cli_config)

    for key, value in optimizer_params.items():
        setattr(callback, key, value)

    for key, value in lr_params.items():
        setattr(callback, key, value)


def find_optimizer_callback(
    callbacks: list[L.pytorch.callbacks.Callback],
) -> SetOptimizerParamsCallback | None:
    """
    Find the SetOptimizerParamsCallback in the list of callbacks.

    Parameters
    ----------
    callbacks : list[L.pytorch.callbacks.Callback]
        List of callbacks to search through.

    Returns
    -------
    SetOptimizerParamsCallback | None
        The SetOptimizerParamsCallback if found, otherwise None.
    """
    for callback in callbacks:
        if isinstance(callback, SetOptimizerParamsCallback):
            return callback

    return None


def load_cli_config(tune_config: TuneConfig) -> dict[str, typing.Any]:
    if not (
        tune_config.cli_config_path.endswith(".yaml")
        or tune_config.cli_config_path.endswith(".yml")
    ):
        msg = f"File must be a YAML file: {tune_config.cli_config_path}"
        raise ValueError(msg)

    if not os.path.exists(tune_config.cli_config_path):
        msg = f"File not found: {tune_config.cli_config_path}"
        raise FileNotFoundError(msg)

    with open(tune_config.cli_config_path, encoding="utf-8") as f:
        cli_config = yaml.full_load(f)

    if "ckpt_path" in cli_config:
        cli_config.pop("ckpt_path")

    apply_key(cli_config, "trainer.max_epochs", tune_config.num_epochs_per_trial)
    apply_key(cli_config, "trainer.enable_progress_bar", value=False)
    # apply_key(
    #     cli_config,
    #     "trainer.callbacks",
    #     [
    #         {
    #             "class_path": "ray.train.lightning.RayTrainReportCallback",
    #         }
    #     ],
    # )
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
    monitor: str,
    grid: dict[str, ray.tune.search.sample.Domain],
    perturbation_interval: int = 10,
    time_attr: str = "training_iteration",
) -> ray.tune.schedulers.TrialScheduler:
    return ray.tune.schedulers.PopulationBasedTraining(
        time_attr=time_attr,
        perturbation_interval=perturbation_interval,
        mode="min",
        hyperparam_mutations=grid,
        metric=monitor,
    )


def configure_tuner(
    tune_config: TuneConfig, cli_config: dict[str, typing.Any]
) -> ray.tune.Tuner:
    reporter = configure_reporter(tune_config.grid, tune_config.metrics)

    if isinstance(tune_config, ASHATuneConfig):
        search_alg = configure_search_alg(tune_config.monitor)
        scheduler = configure_asha_scheduler(
            tune_config.num_epochs_per_trial,
            tune_config.patience,
            tune_config.reduction_factor,
        )
    elif isinstance(tune_config, PBTTuneConfig):
        search_alg = None
        scheduler = configure_pbt_scheduler(
            tune_config.monitor,
            tune_config.grid,
            tune_config.perturbation_interval,
            tune_config.time_attr,
        )
    else:
        msg = f"Unsupported TuneConfig type: {type(tune_config)}"
        raise TypeError(msg)

    ray_tune_config = ray.tune.TuneConfig(
        metric=tune_config.monitor
        if not isinstance(tune_config, PBTTuneConfig)
        else None,
        mode="min" if not isinstance(tune_config, PBTTuneConfig) else None,
        scheduler=scheduler,
        num_samples=tune_config.num_samples,
        max_concurrent_trials=torch.cuda.device_count() * 2
        if torch.cuda.is_available()
        else 4,
        search_alg=search_alg,
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
        )
        if not isinstance(tune_config, PBTTuneConfig)
        else None,
    )

    return ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(
                tune_fn, tune_config=tune_config, cli_config=cli_config
            ),
            resources={"cpu": 4, "gpu": 1},
        ),
        tune_config=ray_tune_config,
        run_config=run_config,
        param_space=tune_config.grid
        if isinstance(tune_config, ASHATuneConfig)
        else None,
    )


def tune(config_path: str) -> ray.tune.ResultGrid:
    """
    Run hyperparameter tuning using unified YAML configuration.

    This function provides a simplified interface for hyperparameter tuning
    that uses a single YAML file containing both the base model configuration
    and the search space specification.

    Parameters
    ----------
    config_path : str
        Path to the unified tune configuration YAML file

    Returns
    -------
    ray.tune.ResultGrid
        The results of the hyperparameter tuning

    Examples
    --------
    >>> from transformertf.utils.tune import tune
    >>> results = tune("unified_tune_config.yml")
    >>> print(f"Best trial config: {results.get_best_result().config}")

    Notes
    -----
    The unified configuration file should contain three main sections:
    - base_config: The base model/data/trainer configuration
    - search_space: Parameter search space specifications
    - tune_config: Ray Tune execution settings (num_samples, metric, logging_metrics, scheduler, etc.)
    """
    from .tune_config import (  # noqa: PLC0415
        create_ray_search_space,
        inject_trial_params,
        load_unified_tune_config,
    )

    # Load and validate the unified configuration
    config = load_unified_tune_config(config_path)

    # Extract configuration sections
    base_config = config["base_config"]
    search_space_config = config["search_space"]
    tune_config = config["tune_config"]

    # Create Ray Tune search space
    ray_search_space = create_ray_search_space(search_space_config)

    # Create tune function that injects trial parameters
    def tune_trial(trial_config: dict) -> None:
        """Single trial function for Ray Tune."""
        # Inject trial parameters into base config
        modified_config = inject_trial_params(base_config, trial_config)

        # Train a single trial with the modified config
        torch.set_float32_matmul_precision("high")

        # Add tune callback to the config
        if "trainer" not in modified_config:
            modified_config["trainer"] = {}
        if "callbacks" not in modified_config["trainer"]:
            modified_config["trainer"]["callbacks"] = []

        # Create tune callback for reporting metrics
        # Combine primary metric with optional logging metrics
        all_metrics = [tune_config["metric"]]
        if "logging_metrics" in tune_config:
            all_metrics.extend(tune_config["logging_metrics"])

        metrics_dict = {metric: metric for metric in all_metrics}

        tune_callback = TuneReportCallback(
            metrics=metrics_dict,
            on="validation_end",
        )
        modified_config["trainer"]["callbacks"].append(tune_callback)

        # Create CLI and run training programmatically
        from .. import main  # noqa: PLC0415

        cli = main.LightningCLI(
            args=modified_config,
            run=False,
            model_class=main.LightningModuleBase,
            datamodule_class=main.DataModuleBase,
            subclass_mode_model=True,
            subclass_mode_data=True,
        )

        # Run training
        cli.trainer.fit(cli.model, cli.datamodule)

    # Set up Ray Tune scheduler
    scheduler_config = tune_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "asha")

    if scheduler_type == "asha":
        scheduler = ray.tune.schedulers.ASHAScheduler(
            max_t=scheduler_config.get("max_t", 100),
            grace_period=scheduler_config.get("grace_period", 10),
            reduction_factor=scheduler_config.get("reduction_factor", 2),
        )
    elif scheduler_type == "pbt":
        scheduler = ray.tune.schedulers.PopulationBasedTraining(
            time_attr="epoch",
            metric=tune_config["metric"],
            mode=tune_config.get("mode", "min"),
            perturbation_interval=scheduler_config.get("perturbation_interval", 20),
            hyperparam_mutations=ray_search_space,
        )
    else:
        scheduler = None

    # Set up search algorithm
    search_config = tune_config.get("search_algorithm", {})
    search_type = search_config.get("type", "basic")

    if search_type == "hyperopt":
        from ray.tune.search.hyperopt import HyperOptSearch  # noqa: PLC0415

        search_alg = HyperOptSearch(
            metric=tune_config["metric"],
            mode=tune_config.get("mode", "min"),
        )
    elif search_type == "optuna":
        from ray.tune.search.optuna import OptunaSearch  # noqa: PLC0415

        search_alg = OptunaSearch(
            metric=tune_config["metric"],
            mode=tune_config.get("mode", "min"),
        )
    else:
        search_alg = None

    # Configure Ray Tune
    tuner = ray.tune.Tuner(
        tune_trial,
        param_space=ray_search_space,
        tune_config=ray.tune.TuneConfig(
            num_samples=tune_config.get("num_samples", 10),
            scheduler=scheduler,
            search_alg=search_alg,
            metric=tune_config["metric"],
            mode=tune_config.get("mode", "min"),
        ),
        run_config=ray.train.RunConfig(
            name=tune_config.get("experiment_name", "unified_tune"),
            storage_path=tune_config.get("storage_path", "./ray_results"),
        ),
    )

    # Run tuning
    results = tuner.fit()

    # Print best result
    best_result = results.get_best_result()
    print(
        f"\nBest trial completed with {tune_config['metric']}: {best_result.metrics[tune_config['metric']]}"
    )
    print(f"Best config: {best_result.config}")

    return results
