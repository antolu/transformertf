"""Tests for transformertf.utils.tune_config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from transformertf.utils.tune_config import (
    TuneConfigValidator,
    create_ray_search_space,
    inject_trial_params,
    load_unified_tune_config,
)


def test_tune_config_validator_init() -> None:
    """Test validator initialization."""
    validator = TuneConfigValidator()
    assert validator is not None


def test_validate_valid_config() -> None:
    """Test validation of a valid configuration."""
    config = {
        "base_config": {"model": {"init_args": {"d_model": 64}}},
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {"num_samples": 10, "metric": "loss/val"},
    }
    validator = TuneConfigValidator()
    validator.validate_config(config)  # Should not raise


def test_validate_valid_config_with_logging_metrics() -> None:
    """Test validation of a valid configuration with logging metrics."""
    config = {
        "base_config": {"model": {"init_args": {"d_model": 64}}},
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 10,
            "metric": "RMSE/val",
            "logging_metrics": ["MSE/val", "SMAPE/val", "loss/train"],
        },
    }
    validator = TuneConfigValidator()
    validator.validate_config(config)  # Should not raise


def test_validate_missing_base_config() -> None:
    """Test validation fails when base_config is missing."""
    config = {
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing required section 'base_config'"):
        validator.validate_config(config)


def test_validate_missing_search_space() -> None:
    """Test validation fails when search_space is missing."""
    config = {
        "base_config": {"model": {}},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing required section 'search_space'"):
        validator.validate_config(config)


def test_validate_missing_tune_config() -> None:
    """Test validation fails when tune_config is missing."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing required section 'tune_config'"):
        validator.validate_config(config)


def test_validate_empty_search_space() -> None:
    """Test validation fails for empty search space."""
    config = {
        "base_config": {"model": {}},
        "search_space": {},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Search space cannot be empty"):
        validator.validate_config(config)


def test_validate_invalid_search_param_type() -> None:
    """Test validation fails for invalid search parameter types."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": "not_a_dict"},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(TypeError, match="must be a dictionary"):
        validator.validate_config(config)


def test_validate_missing_type_field() -> None:
    """Test validation fails when 'type' field is missing."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"values": [1, 2, 3]}},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing 'type' field for parameter 'param'"):
        validator.validate_config(config)


def test_validate_invalid_sampling_type() -> None:
    """Test validation fails for invalid sampling types."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "invalid_type", "values": [1, 2]}},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Invalid sampling type 'invalid_type'"):
        validator.validate_config(config)


@pytest.mark.parametrize(
    ("sampling_config", "expected_error"),
    [
        # uniform with invalid range
        ({"type": "uniform", "min": 5, "max": 2}, "min \\(5\\) must be < max \\(2\\)"),
        # loguniform with invalid range
        (
            {"type": "loguniform", "min": 1e-2, "max": 1e-3},
            "min \\(0.01\\) must be < max \\(0.001\\)",
        ),
        # choice with empty values
        (
            {"type": "choice", "values": []},
            "'values' for choice .* must be a non-empty list",
        ),
        # choice with non-list values
        (
            {"type": "choice", "values": "not_a_list"},
            "'values' for choice .* must be a non-empty list",
        ),
        # randint with invalid range
        (
            {"type": "randint", "lower": 10, "upper": 5},
            "lower \\(10\\) must be < upper \\(5\\)",
        ),
        # lograndint with invalid range
        (
            {"type": "lograndint", "lower": 100, "upper": 10},
            "lower \\(100\\) must be < upper \\(10\\)",
        ),
    ],
)
def test_validate_sampling_constraints(
    sampling_config: dict, expected_error: str
) -> None:
    """Test validation of specific sampling type constraints."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": sampling_config},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match=expected_error):
        validator.validate_config(config)


def test_validate_missing_required_fields() -> None:
    """Test validation fails when required fields are missing."""
    # uniform missing max
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "uniform", "min": 0}},
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing required field 'max' for uniform"):
        validator.validate_config(config)


def test_validate_tune_config_missing_num_samples() -> None:
    """Test validation fails when tune_config missing num_samples."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {"metric": "loss"},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing required field 'num_samples'"):
        validator.validate_config(config)


def test_validate_tune_config_missing_metric() -> None:
    """Test validation fails when tune_config missing metric."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {"num_samples": 10},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="Missing required field 'metric'"):
        validator.validate_config(config)


def test_validate_invalid_metric_not_string() -> None:
    """Test validation fails when metric is not a string."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {"num_samples": 10, "metric": 123},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validator.validate_config(config)


def test_validate_invalid_metric_empty_string() -> None:
    """Test validation fails when metric is empty string."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {"num_samples": 10, "metric": ""},
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validator.validate_config(config)


def test_validate_invalid_logging_metrics_not_list() -> None:
    """Test validation fails when logging_metrics is not a list."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {
            "num_samples": 10,
            "metric": "loss",
            "logging_metrics": "not_a_list",
        },
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="must be a list"):
        validator.validate_config(config)


def test_validate_invalid_logging_metrics_empty_string() -> None:
    """Test validation fails when logging_metrics contains empty string."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {
            "num_samples": 10,
            "metric": "loss",
            "logging_metrics": ["MSE", "", "RMSE"],
        },
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validator.validate_config(config)


def test_validate_invalid_logging_metrics_non_string() -> None:
    """Test validation fails when logging_metrics contains non-string."""
    config = {
        "base_config": {"model": {}},
        "search_space": {"param": {"type": "choice", "values": [1, 2]}},
        "tune_config": {
            "num_samples": 10,
            "metric": "loss",
            "logging_metrics": ["MSE", 123, "RMSE"],
        },
    }
    validator = TuneConfigValidator()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validator.validate_config(config)


def test_inject_simple_param() -> None:
    """Test injection of a simple parameter."""
    base_config = {"model": {"init_args": {"d_model": 64}}}
    trial_params = {"model.init_args.d_model": 128}
    result = inject_trial_params(base_config, trial_params)

    assert result["model"]["init_args"]["d_model"] == 128
    # Ensure original config is not mutated
    assert base_config["model"]["init_args"]["d_model"] == 64


def test_inject_multiple_params() -> None:
    """Test injection of multiple parameters."""
    base_config = {
        "model": {"init_args": {"d_model": 64, "num_heads": 4}},
        "optimizer": {"init_args": {"lr": 0.001}},
    }
    trial_params = {
        "model.init_args.d_model": 128,
        "model.init_args.num_heads": 8,
        "optimizer.init_args.lr": 0.01,
    }
    result = inject_trial_params(base_config, trial_params)

    assert result["model"]["init_args"]["d_model"] == 128
    assert result["model"]["init_args"]["num_heads"] == 8
    assert result["optimizer"]["init_args"]["lr"] == 0.01


def test_inject_new_path() -> None:
    """Test injection creates new nested paths."""
    base_config = {"model": {"init_args": {}}}
    trial_params = {"model.init_args.new_param": 42}
    result = inject_trial_params(base_config, trial_params)

    assert result["model"]["init_args"]["new_param"] == 42


def test_inject_completely_new_path() -> None:
    """Test injection creates completely new nested structure."""
    base_config = {}
    trial_params = {"new.nested.path": "value"}
    result = inject_trial_params(base_config, trial_params)

    assert result["new"]["nested"]["path"] == "value"


def test_inject_deep_nesting() -> None:
    """Test injection with deep nesting."""
    base_config = {}
    trial_params = {"a.b.c.d.e.f": "deep_value"}
    result = inject_trial_params(base_config, trial_params)

    assert result["a"]["b"]["c"]["d"]["e"]["f"] == "deep_value"


def test_inject_overwrites_existing() -> None:
    """Test injection overwrites existing values."""
    base_config = {"existing": {"value": "old"}}
    trial_params = {"existing.value": "new"}
    result = inject_trial_params(base_config, trial_params)

    assert result["existing"]["value"] == "new"


def test_inject_various_types() -> None:
    """Test injection with various parameter types."""
    base_config = {}
    trial_params = {
        "int_param": 42,
        "float_param": 3.14,
        "str_param": "hello",
        "bool_param": True,
        "list_param": [1, 2, 3],
        "dict_param": {"nested": "value"},
    }
    result = inject_trial_params(base_config, trial_params)

    assert result["int_param"] == 42
    assert result["float_param"] == 3.14
    assert result["str_param"] == "hello"
    assert result["bool_param"] is True
    assert result["list_param"] == [1, 2, 3]
    assert result["dict_param"] == {"nested": "value"}


def test_create_ray_search_space_choice() -> None:
    """Test choice sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "choice", "values": [1, 2, 3, "test"]}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Categorical)


def test_create_ray_search_space_uniform() -> None:
    """Test uniform sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "uniform", "min": 0.1, "max": 1.0}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_loguniform() -> None:
    """Test loguniform sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "loguniform", "min": 1e-5, "max": 1e-2}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_loguniform_with_base() -> None:
    """Test loguniform sampling with custom base."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {
        "param": {"type": "loguniform", "min": 1e-3, "max": 1e-1, "base": 2}
    }
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_normal() -> None:
    """Test normal sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "normal", "mean": 0.0, "sd": 1.0}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_lognormal() -> None:
    """Test lognormal sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "lognormal", "mean": 0.0, "sd": 1.0}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_randint() -> None:
    """Test randint sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "randint", "lower": 1, "upper": 10}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Integer)


def test_create_ray_search_space_randint_with_step() -> None:
    """Test randint sampling with custom step."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "randint", "lower": 0, "upper": 100, "step": 10}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Integer)


def test_create_ray_search_space_randn() -> None:
    """Test randn sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "randn"}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_randn_with_params() -> None:
    """Test randn sampling with custom mean and sd."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "randn", "mean": 5.0, "sd": 2.0}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Float)


def test_create_ray_search_space_lograndint() -> None:
    """Test lograndint sampling space creation."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {"param": {"type": "lograndint", "lower": 1, "upper": 1000}}
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Integer)


def test_create_ray_search_space_lograndint_with_base() -> None:
    """Test lograndint sampling with custom base."""
    pytest.importorskip("ray.tune")
    import ray.tune

    search_config = {
        "param": {"type": "lograndint", "lower": 1, "upper": 256, "base": 2}
    }
    result = create_ray_search_space(search_config)

    assert "param" in result
    assert isinstance(result["param"], ray.tune.sample.Integer)


def test_create_ray_search_space_multiple_params() -> None:
    """Test multiple parameters in search space."""
    pytest.importorskip("ray.tune")

    search_config = {
        "model.d_model": {"type": "choice", "values": [64, 128, 256]},
        "optimizer.lr": {"type": "loguniform", "min": 1e-5, "max": 1e-2},
        "data.batch_size": {"type": "randint", "lower": 16, "upper": 128},
    }
    result = create_ray_search_space(search_config)

    assert len(result) == 3
    assert "model.d_model" in result
    assert "optimizer.lr" in result
    assert "data.batch_size" in result


def test_create_ray_search_space_unsupported_type() -> None:
    """Test error for unsupported sampling type."""
    search_config = {"param": {"type": "unsupported_type"}}
    with pytest.raises(ValueError, match="Unsupported sampling type: unsupported_type"):
        create_ray_search_space(search_config)


def test_load_unified_tune_config_valid() -> None:
    """Test loading a valid configuration file."""
    config_data = {
        "base_config": {"model": {"init_args": {"d_model": 64}}},
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [64, 128]}
        },
        "tune_config": {"num_samples": 10, "metric": "loss/val"},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        result = load_unified_tune_config(temp_path)
        assert result == config_data
    finally:
        Path(temp_path).unlink()


def test_load_unified_tune_config_valid_with_logging_metrics() -> None:
    """Test loading a valid configuration file with logging metrics."""
    config_data = {
        "base_config": {"model": {"init_args": {"d_model": 64}}},
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [64, 128]}
        },
        "tune_config": {
            "num_samples": 10,
            "metric": "RMSE/val",
            "logging_metrics": ["MSE/val", "SMAPE/val"],
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        result = load_unified_tune_config(temp_path)
        assert result == config_data
    finally:
        Path(temp_path).unlink()


def test_load_unified_tune_config_nonexistent() -> None:
    """Test loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_unified_tune_config("nonexistent_file.yml")


def test_load_unified_tune_config_empty() -> None:
    """Test loading an empty file raises ValueError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        f.write("")  # Empty file
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Empty or invalid YAML file"):
            load_unified_tune_config(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_unified_tune_config_invalid_yaml() -> None:
    """Test loading invalid YAML raises an error."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        f.write("invalid: yaml: content: [")  # Invalid YAML
        temp_path = f.name

    try:
        with pytest.raises(yaml.YAMLError):
            load_unified_tune_config(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_unified_tune_config_validation_error() -> None:
    """Test loading config that fails validation."""
    config_data = {
        "base_config": {"model": {}},
        "search_space": {},  # Empty search space - should fail validation
        "tune_config": {"num_samples": 10, "metric": "loss"},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Search space cannot be empty"):
            load_unified_tune_config(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_unified_tune_config_pathlib() -> None:
    """Test loading config using pathlib.Path object."""
    config_data = {
        "base_config": {"model": {"init_args": {"d_model": 64}}},
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [64, 128]}
        },
        "tune_config": {"num_samples": 10, "metric": "loss/val"},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)

    try:
        result = load_unified_tune_config(temp_path)
        assert result == config_data
    finally:
        temp_path.unlink()
