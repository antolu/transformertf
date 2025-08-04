"""
Configuration system for YAML-based hyperparameter tuning.

This module provides validation and parameter injection utilities for
unified YAML-based tune configurations that combine base model configs
with search space specifications.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

# Ray Tune import with error handling
try:
    import ray.tune
except ImportError:
    print("ERROR: Ray Tune is required for hyperparameter tuning!")
    print("Please install it with: pip install ray[tune]")
    sys.exit(1)

__all__ = [
    "TuneConfigValidator",
    "create_ray_search_space",
    "inject_trial_params",
    "load_unified_tune_config",
]

# Valid Ray Tune sampling types and their required/optional parameters
VALID_SAMPLING_TYPES = {
    "choice": {"required": ["values"], "optional": []},
    "uniform": {"required": ["min", "max"], "optional": []},
    "loguniform": {"required": ["min", "max"], "optional": ["base"]},
    "normal": {"required": ["mean", "sd"], "optional": []},
    "lognormal": {"required": ["mean", "sd"], "optional": []},
    "randint": {"required": ["lower", "upper"], "optional": ["step"]},
    "randn": {"required": [], "optional": ["mean", "sd"]},
    "lograndint": {"required": ["lower", "upper"], "optional": ["base"]},
}


class TuneConfigValidator:
    """Validator for unified tune configuration files."""

    def __init__(self) -> None:
        """Initialize the validator."""

    def validate_config(self, config: dict) -> None:
        """
        Validate a unified tune configuration.

        Parameters
        ----------
        config : dict
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


def inject_trial_params(base_config: dict, trial_params: dict) -> dict:
    """
    Inject trial parameters into base config using dot notation.

    Parameters
    ----------
    base_config : dict
        The base configuration dictionary
    trial_params : dict
        Trial parameters with dot-notation keys (e.g., "model.init_args.num_layers")

    Returns
    -------
    dict
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


def create_ray_search_space(search_space_config: dict) -> dict:
    """
    Convert search space configuration to Ray Tune search space.

    Parameters
    ----------
    search_space_config : dict
        Search space configuration from YAML

    Returns
    -------
    dict
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
        elif sampling_type == "normal":
            ray_search_space[param_path] = ray.tune.normal(
                param_config["mean"], param_config["sd"]
            )
        elif sampling_type == "lognormal":
            ray_search_space[param_path] = ray.tune.lognormal(
                param_config["mean"], param_config["sd"]
            )
        elif sampling_type == "randint":
            step = param_config.get("step", 1)
            ray_search_space[param_path] = ray.tune.randint(
                param_config["lower"], param_config["upper"], step=step
            )
        elif sampling_type == "randn":
            mean = param_config.get("mean", 0.0)
            sd = param_config.get("sd", 1.0)
            ray_search_space[param_path] = ray.tune.randn(mean=mean, sd=sd)
        elif sampling_type == "lograndint":
            base = param_config.get("base", 10)
            ray_search_space[param_path] = ray.tune.lograndint(
                param_config["lower"], param_config["upper"], base=base
            )
        else:
            msg = f"Unsupported sampling type: {sampling_type}"
            raise ValueError(msg)

    return ray_search_space


def load_unified_tune_config(config_path: str | Path) -> dict:
    """
    Load and validate a unified tune configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file

    Returns
    -------
    dict
        Loaded and validated configuration

    Raises
    ------
    ValueError
        If the configuration is invalid
    FileNotFoundError
        If the configuration file doesn't exist
    """
    config_path = Path(config_path)

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
