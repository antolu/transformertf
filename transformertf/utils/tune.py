"""
Utilities for implementing hyperparameter tuning with Ray Tune.

Examples
--------
The following example demonstrates how to use the `tune` module to implement
hyperparameter tuning with Ray Tune.
"""

from __future__ import annotations

import logging
import os

import lightning as L
import ray
import ray.train
import ray.tune
import ray.tune.integration.pytorch_lightning
import ray.tune.schedulers
import ray.tune.search
import torch

log = logging.getLogger(__name__)


__all__ = [
    "TuneReportCallback",
    "tune",
]


# Legacy dataclasses removed - use YAML configuration instead


class TuneReportCallback(
    ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback,
    L.pytorch.callbacks.Callback,
): ...


# Old tune implementation removed - use YAML-based tune() function instead


def tune(config_path: str) -> ray.tune.ResultGrid:
    """
    Run hyperparameter tuning using YAML configuration.

    This function provides a comprehensive interface for hyperparameter tuning
    that uses a single YAML file containing both the base model configuration
    and the search space specification.

    Parameters
    ----------
    config_path : str
        Path to the tune configuration YAML file

    Returns
    -------
    ray.tune.ResultGrid
        The results of the hyperparameter tuning

    Examples
    --------
    >>> from transformertf.utils.tune import tune
    >>> results = tune("tune_config.yml")
    >>> print(f"Best trial config: {results.get_best_result().config}")

    Notes
    -----
    The configuration file should contain three main sections:
    - base_config: The base model/data/trainer configuration
    - search_space: Parameter search space specifications
    - tune_config: Ray Tune execution settings (num_samples, metric, logging_metrics, scheduler, etc.)

    Environment variable propagation to Ray workers can be configured via:
    tune_config:
      env_vars:
        patterns:
          - "NEPTUNE_*"      # All Neptune environment variables
          - "WANDB_*"        # All Weights & Biases variables
          - "*_proxy"        # All lowercase proxy variables
          - "*_PROXY"        # All uppercase proxy variables
          - "AWS_*"          # All AWS credentials
          - "CUSTOM_API_KEY" # Specific exact variable name
    """
    from .tune_config import (  # noqa: PLC0415
        create_ray_search_space,
        inject_trial_params,
        load_tune_config,
    )

    # Load and validate the configuration
    config = load_tune_config(config_path)

    # Extract configuration sections
    base_config = config["base_config"]
    search_space_config = config["search_space"]
    tune_config = config["tune_config"]

    # Initialize Ray with proper GPU detection and environment variable propagation
    if not ray.is_initialized():
        # Get environment variable patterns from config
        env_patterns = tune_config.get("env_vars", {}).get("patterns", [])

        # Collect environment variables matching the patterns
        env_vars = {}
        for pattern in env_patterns:
            if pattern.endswith("*"):
                # Prefix pattern matching
                prefix = pattern[:-1]
                env_vars.update({
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith(prefix)
                })
            elif pattern.startswith("*"):
                # Suffix pattern matching
                suffix = pattern[1:]
                env_vars.update({
                    key: value
                    for key, value in os.environ.items()
                    if key.endswith(suffix)
                })
            else:
                # Exact match
                if pattern in os.environ:
                    env_vars[pattern] = os.environ[pattern]

        # Initialize Ray with environment variables if any were found
        if env_vars:
            runtime_env = {"env_vars": env_vars}
            ray.init(runtime_env=runtime_env)
        else:
            ray.init()

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

        # Configure trainer for Ray environment
        if "trainer" not in modified_config:
            modified_config["trainer"] = {}

        # Set up proper GPU/CPU configuration for Ray workers
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            modified_config["trainer"]["accelerator"] = "gpu"
            modified_config["trainer"]["devices"] = 1  # Ray allocates 1 GPU per worker
        else:
            modified_config["trainer"]["accelerator"] = "cpu"
            modified_config["trainer"]["devices"] = 1

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
    elif search_type == "ax":
        try:
            from ray.tune.search.ax import AxSearch  # noqa: PLC0415

            # Ax-specific parameters
            ax_config = {
                "metric": tune_config["metric"],
                "mode": tune_config.get("mode", "min"),
            }

            # Optional Ax parameters to pass to AxClient via ax_kwargs
            # Only include parameters that are actually supported
            supported_ax_params = [
                "verbose_logging"
            ]  # Add other supported params as needed

            for param in supported_ax_params:
                if param in search_config:
                    ax_config[param] = search_config[param]

            search_alg = AxSearch(**ax_config)
        except (ImportError, AssertionError, TypeError) as e:
            msg = "Ax is required for 'ax' search algorithm. Install with: pip install ax-platform"
            raise ImportError(msg) from e

    elif search_type == "bayesopt":
        try:
            from ray.tune.search.bayesopt import BayesOptSearch  # noqa: PLC0415

            # BayesOpt-specific parameters
            bayesopt_config = {
                "metric": tune_config["metric"],
                "mode": tune_config.get("mode", "min"),
            }

            # Optional BayesOpt parameters
            if "utility_kwargs" in search_config:
                bayesopt_config["utility_kwargs"] = search_config["utility_kwargs"]
            if "random_state" in search_config:
                bayesopt_config["random_state"] = search_config["random_state"]
            if "random_search_steps" in search_config:
                bayesopt_config["random_search_steps"] = search_config[
                    "random_search_steps"
                ]

            search_alg = BayesOptSearch(**bayesopt_config)
        except (ImportError, AssertionError) as e:
            msg = "BayesOpt is required for 'bayesopt' search algorithm. Install with: pip install bayesian-optimization"
            raise ImportError(msg) from e

    elif search_type == "nevergrad":
        try:
            from ray.tune.search.nevergrad import NevergradSearch  # noqa: PLC0415

            # Nevergrad-specific parameters
            nevergrad_config = {
                "metric": tune_config["metric"],
                "mode": tune_config.get("mode", "min"),
            }

            # Optional Nevergrad parameters
            if "optimizer" in search_config:
                nevergrad_config["optimizer"] = search_config["optimizer"]
            # Note: Budget is typically handled by the Tuner, not the search algorithm

            search_alg = NevergradSearch(**nevergrad_config)
        except (ImportError, AssertionError) as e:
            msg = "Nevergrad is required for 'nevergrad' search algorithm. Install with: pip install nevergrad"
            raise ImportError(msg) from e

    else:
        search_alg = None

    # Get resource configuration
    resources = tune_config.get("resources", {"cpu": 4, "gpu": 1})

    # Ensure GPU count doesn't exceed available GPUs
    if "gpu" in resources and resources["gpu"] > 0:
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            resources["gpu"] = 0
        elif resources["gpu"] > available_gpus:
            resources["gpu"] = available_gpus

    # Configure Ray Tune with proper resource allocation
    tuner = ray.tune.Tuner(
        ray.tune.with_resources(tune_trial, resources=resources),
        param_space=ray_search_space,
        tune_config=ray.tune.TuneConfig(
            num_samples=tune_config.get("num_samples", 10),
            scheduler=scheduler,
            search_alg=search_alg,
            metric=tune_config["metric"],
            mode=tune_config.get("mode", "min"),
        ),
        run_config=ray.train.RunConfig(
            name=tune_config.get("experiment_name", "tune_experiment"),
            storage_path=os.path.abspath(
                tune_config.get("storage_path", "./ray_results")
            ),
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
