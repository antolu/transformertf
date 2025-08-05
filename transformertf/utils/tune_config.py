"""
Configuration system for YAML-based hyperparameter tuning.

This module provides validation and parameter injection utilities for
unified YAML-based tune configurations that combine base model configs
with search space specifications.
"""

from __future__ import annotations

import copy
import os
import pathlib
import sys
import typing

import yaml

# Ray Tune import with error handling
try:
    import ray.tune
except ImportError:
    print("ERROR: Ray Tune is required for hyperparameter tuning!")
    print("Please install it with: pip install ray[tune]")
    sys.exit(1)

__all__ = [
    "EnvVarsConfig",
    "ResourceConfig",
    "SchedulerConfig",
    "SearchAlgorithmConfig",
    "SearchSpaceParam",
    "TuneConfig",
    "TuneConfigValidator",
    "TunerConfig",
    "create_ray_search_space",
    "inject_trial_params",
    "load_tune_config",
]

# Valid Ray Tune sampling types and their required/optional parameters
# TypedDict definitions for tune configuration structure


class ResourceConfig(typing.TypedDict, total=False):
    """Resource allocation configuration."""

    cpu: int | float
    """Number of CPU cores to allocate per trial"""
    gpu: int | float
    """Number of GPU devices to allocate per trial"""
    memory: int | str
    """Memory allocation per trial (bytes or string like '2GB')"""


class SchedulerConfig(typing.TypedDict, total=False):
    """Scheduler configuration."""

    type: typing.Literal["asha", "pbt", "median_stopping", "hyperband"]
    """Type of scheduler to use for trial management"""
    max_t: int
    """Maximum number of epochs/iterations per trial"""
    grace_period: int
    """Minimum number of epochs before early stopping"""
    reduction_factor: int | float
    """Factor by which to reduce number of trials at each rung"""
    perturbation_interval: int
    """Frequency of parameter perturbation for PBT"""
    time_attr: str
    """Time attribute to use for scheduling decisions"""


class SearchAlgorithmConfig(typing.TypedDict, total=False):
    """Search algorithm configuration."""

    type: typing.Literal["hyperopt", "ax", "bayesopt", "nevergrad", "optuna"]
    """Type of search algorithm to use"""
    num_bootstrap: int
    """Number of bootstrap samples for Ax"""
    min_trials_observed: int
    """Minimum trials before using model for Ax"""
    verbose_logging: bool
    """Enable verbose logging for search algorithm"""
    random_state: int
    """Random seed for reproducible results"""
    random_search_steps: int
    """Number of random search steps before optimization"""
    utility_kwargs: dict[str, typing.Any]
    """Additional arguments for utility function"""
    optimizer: str
    """Optimizer name for Nevergrad"""


class EnvVarsConfig(typing.TypedDict, total=False):
    """Environment variables configuration."""

    patterns: list[str]
    """List of environment variable patterns to propagate"""


class TunerConfig(typing.TypedDict):
    """Tune configuration section."""

    num_samples: int
    """Number of trials to run"""
    metric: str
    """Metric name to optimize"""
    mode: typing.NotRequired[typing.Literal["min", "max"]]
    """Whether to minimize or maximize the metric"""
    resources: typing.NotRequired[ResourceConfig]
    """Resource allocation per trial"""
    scheduler: typing.NotRequired[SchedulerConfig]
    """Trial scheduler configuration"""
    search_algorithm: typing.NotRequired[SearchAlgorithmConfig]
    """Search algorithm configuration"""
    experiment_name: typing.NotRequired[str]
    """Name of the experiment"""
    storage_path: typing.NotRequired[str]
    """Path to store experiment results"""
    logging_metrics: typing.NotRequired[list[str]]
    """Additional metrics to log"""
    env_vars: typing.NotRequired[EnvVarsConfig]
    """Environment variables to propagate"""


class SearchSpaceParam(typing.TypedDict):
    """Individual search space parameter configuration."""

    type: typing.Literal[
        "choice",
        "uniform",
        "loguniform",
        "randint",
        "randn",
        "lograndint",
        "grid_search",
    ]
    """Type of sampling distribution"""
    values: typing.NotRequired[list[typing.Any]]
    """List of values for choice and grid_search"""
    min: typing.NotRequired[float]
    """Minimum value for uniform and loguniform"""
    max: typing.NotRequired[float]
    """Maximum value for uniform and loguniform"""
    lower: typing.NotRequired[int]
    """Lower bound for integer distributions"""
    upper: typing.NotRequired[int]
    """Upper bound for integer distributions"""
    base: typing.NotRequired[float]
    """Base for logarithmic distributions"""
    mean: typing.NotRequired[float]
    """Mean for normal distribution"""
    std: typing.NotRequired[float]
    """Standard deviation for normal distribution"""


class TuneConfig(typing.TypedDict):
    """Complete unified tune configuration structure."""

    base_config: dict[str, typing.Any]
    """Base Lightning configuration"""
    search_space: dict[str, SearchSpaceParam]
    """Hyperparameter search space definition"""
    tune_config: TunerConfig
    """Ray Tune configuration options"""


VALID_SAMPLING_TYPES = {
    "choice": {"required": ["values"], "optional": []},
    "uniform": {"required": ["min", "max"], "optional": []},
    "loguniform": {"required": ["min", "max"], "optional": ["base"]},
    "randint": {"required": ["lower", "upper"], "optional": []},
    "randn": {"required": [], "optional": ["mean", "std"]},
    "lograndint": {"required": ["lower", "upper"], "optional": ["base"]},
    "grid_search": {"required": ["values"], "optional": []},
}


class TuneConfigValidator:
    """Validator for unified tune configuration files."""

    def __init__(self) -> None:
        """Initialize the validator."""

    def validate_config(self, config: TuneConfig) -> None:
        """
        Validate a unified tune configuration.

        Parameters
        ----------
        config : TuneConfig
            The loaded configuration dictionary

        Raises
        ------
        ValueError
            If the configuration is invalid
        """
        # Validate top-level structure
        required_sections = ["base_config", "search_space", "tune_config"]
        for section in required_sections:
            if section not in config:
                msg = f"Missing required section '{section}' in tune configuration"
                raise ValueError(msg)

        # Validate search space
        self._validate_search_space(config["search_space"])

        # Validate tune config
        self._validate_tune_config(config["tune_config"])

    def _validate_search_space(self, search_space: dict) -> None:
        """Validate search space configuration."""
        if not search_space:
            msg = "Search space cannot be empty"
            raise ValueError(msg)

        for param_path, param_config in search_space.items():
            if not isinstance(param_config, dict):
                msg = f"Search space parameter '{param_path}' must be a dictionary"
                raise TypeError(msg)

            if "type" not in param_config:
                msg = f"Missing 'type' field for parameter '{param_path}'"
                raise ValueError(msg)

            sampling_type = param_config["type"]
            if sampling_type not in VALID_SAMPLING_TYPES:
                msg = f"Invalid sampling type '{sampling_type}' for parameter '{param_path}'. "
                msg += f"Valid types: {list(VALID_SAMPLING_TYPES.keys())}"
                raise ValueError(msg)

            # Check required fields
            required_fields = VALID_SAMPLING_TYPES[sampling_type]["required"]
            for field in required_fields:
                if field not in param_config:
                    msg = f"Missing required field '{field}' for {sampling_type} in parameter '{param_path}'"
                    raise ValueError(msg)

            # Validate specific constraints
            self._validate_sampling_constraints(param_path, param_config)

    def _validate_sampling_constraints(self, param_path: str, config: dict) -> None:
        """Validate constraints for specific sampling types."""
        sampling_type = config["type"]

        if sampling_type in ["uniform", "loguniform"]:
            min_val = config["min"]
            max_val = config["max"]
            if min_val >= max_val:
                msg = f"Invalid range for {sampling_type} in '{param_path}': min ({min_val}) must be < max ({max_val})"
                raise ValueError(msg)

        elif sampling_type == "choice":
            values = config["values"]
            if not isinstance(values, list) or len(values) == 0:
                msg = f"'values' for choice in '{param_path}' must be a non-empty list"
                raise ValueError(msg)

        elif sampling_type in ["randint", "lograndint"]:
            lower = config["lower"]
            upper = config["upper"]
            if lower >= upper:
                msg = f"Invalid range for {sampling_type} in '{param_path}': lower ({lower}) must be < upper ({upper})"
                raise ValueError(msg)

    def _validate_tune_config(self, tune_config: dict) -> None:
        """Validate tune configuration section."""
        required_fields = ["num_samples", "metric"]
        for field in required_fields:
            if field not in tune_config:
                msg = f"Missing required field '{field}' in tune_config"
                raise ValueError(msg)

        # Validate metric
        metric = tune_config["metric"]
        if not isinstance(metric, str) or not metric.strip():
            msg = "tune_config.metric must be a non-empty string"
            raise ValueError(msg)

        # Validate optional logging_metrics
        if "logging_metrics" in tune_config:
            logging_metrics = tune_config["logging_metrics"]
            if not isinstance(logging_metrics, list):
                msg = "tune_config.logging_metrics must be a list"
                raise ValueError(msg)

            for i, log_metric in enumerate(logging_metrics):
                if not isinstance(log_metric, str) or not log_metric.strip():
                    msg = f"tune_config.logging_metrics[{i}] must be a non-empty string"
                    raise ValueError(msg)


def inject_trial_params(
    base_config: dict[str, typing.Any], trial_params: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """
    Inject trial parameters into base config using dot notation.

    Parameters
    ----------
    base_config : dict[str, Any]
        The base configuration dictionary
    trial_params : dict[str, Any]
        Trial parameters with dot-notation keys (e.g., "model.init_args.num_layers")

    Returns
    -------
    dict[str, Any]
        Modified configuration with trial parameters injected

    Examples
    --------
    >>> base_config = {"model": {"init_args": {"num_layers": 2}}}
    >>> trial_params = {"model.init_args.num_layers": 4}
    >>> result = inject_trial_params(base_config, trial_params)
    >>> result["model"]["init_args"]["num_layers"]
    4
    """
    config = copy.deepcopy(base_config)

    for param_path, value in trial_params.items():
        # Split dot notation path
        keys = param_path.split(".")

        # Navigate to parent dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    return config


def create_ray_search_space(
    search_space_config: dict[str, SearchSpaceParam],
) -> dict[str, typing.Any]:
    """
    Convert search space configuration to Ray Tune search space.

    Parameters
    ----------
    search_space_config : dict[str, SearchSpaceParam]
        Search space configuration from YAML

    Returns
    -------
    dict[str, Any]
        Ray Tune search space dictionary

    Examples
    --------
    >>> config = {
    ...     "model.num_layers": {"type": "choice", "values": [1, 2, 3]},
    ...     "optimizer.lr": {"type": "loguniform", "min": 1e-5, "max": 1e-2}
    ... }
    >>> search_space = create_ray_search_space(config)
    >>> isinstance(search_space["model.num_layers"], ray.tune.sample.Categorical)
    True
    """
    ray_search_space = {}

    for param_path, param_config in search_space_config.items():
        sampling_type = param_config["type"]

        if sampling_type == "choice":
            ray_search_space[param_path] = ray.tune.choice(param_config["values"])
        elif sampling_type == "uniform":
            ray_search_space[param_path] = ray.tune.uniform(
                param_config["min"], param_config["max"]
            )
        elif sampling_type == "loguniform":
            base = param_config.get("base", 10)
            ray_search_space[param_path] = ray.tune.loguniform(
                param_config["min"], param_config["max"], base=base
            )
        elif sampling_type == "randint":
            ray_search_space[param_path] = ray.tune.randint(
                param_config["lower"], param_config["upper"]
            )
        elif sampling_type == "randn":
            mean = param_config.get("mean", 0.0)
            std = param_config.get("std", 1.0)
            ray_search_space[param_path] = ray.tune.randn(mean=mean, sd=std)
        elif sampling_type == "lograndint":
            base = param_config.get("base", 10)
            ray_search_space[param_path] = ray.tune.lograndint(
                param_config["lower"], param_config["upper"], base=base
            )
        elif sampling_type == "grid_search":
            ray_search_space[param_path] = ray.tune.grid_search(param_config["values"])
        else:
            msg = f"Unsupported sampling type: {sampling_type}"
            raise ValueError(msg)

    return ray_search_space


def load_tune_config(config_path: str | pathlib.Path) -> TuneConfig:
    """
    Load and validate a unified tune configuration file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the YAML configuration file

    Returns
    -------
    TuneConfig
        Loaded and validated configuration

    Raises
    ------
    ValueError
        If the configuration is invalid
    FileNotFoundError
        If the configuration file doesn't exist
    """
    config_path = pathlib.Path(os.path.abspath(config_path))

    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if not config:
        msg = f"Empty or invalid YAML file: {config_path}"
        raise ValueError(msg)

    # Validate configuration
    validator = TuneConfigValidator()
    validator.validate_config(config)

    return config
