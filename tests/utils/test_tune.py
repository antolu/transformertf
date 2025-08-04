"""Tests for transformertf.utils.tune module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from transformertf.utils.tune import (
    ASHATuneConfig,
    PBTTuneConfig,
    TuneConfig,
    apply_config,
    apply_key,
    find_lr_scheduler_params,
    find_optimizer_params,
    load_cli_config,
    make_set_optimizer_params_callback,
    make_tune_callback,
    read_from_dot_key,
)


def test_tune_config_init() -> None:
    """Test TuneConfig initialization."""
    config = TuneConfig(
        cli_config_path="/path/to/config.yml",
        grid={"lr": Mock()},
        param2key={"lr": "optimizer.init_args.lr"},
        metrics=["loss", "accuracy"],
        monitor="loss",
        log_dir="/tmp/logs",
        run_name="test_run",
        num_epochs_per_trial=10,
        num_samples=5,
    )
    assert config.cli_config_path == "/path/to/config.yml"
    assert config.monitor == "loss"
    assert config.num_samples == 5


def test_tune_config_monitor_validation() -> None:
    """Test TuneConfig validates monitor is in metrics."""
    with pytest.raises(ValueError, match="Monitor metric .* not found in metrics"):
        TuneConfig(
            cli_config_path="/path/to/config.yml",
            grid={"lr": Mock()},
            param2key={"lr": "optimizer.init_args.lr"},
            metrics=["loss", "accuracy"],
            monitor="rmse",  # Not in metrics
            log_dir="/tmp/logs",
            run_name="test_run",
            num_epochs_per_trial=10,
            num_samples=5,
        )


def test_asha_tune_config_init() -> None:
    """Test ASHATuneConfig initialization with defaults."""
    config = ASHATuneConfig(
        cli_config_path="/path/to/config.yml",
        grid={"lr": Mock()},
        param2key={"lr": "optimizer.init_args.lr"},
        metrics=["loss"],
        monitor="loss",
        log_dir="/tmp/logs",
        run_name="test_run",
        num_epochs_per_trial=10,
        num_samples=5,
    )
    assert config.patience == 10
    assert config.reduction_factor == 2


def test_pbt_tune_config_init() -> None:
    """Test PBTTuneConfig initialization with defaults."""
    config = PBTTuneConfig(
        cli_config_path="/path/to/config.yml",
        grid={"lr": Mock()},
        param2key={"lr": "optimizer.init_args.lr"},
        metrics=["loss"],
        monitor="loss",
        log_dir="/tmp/logs",
        run_name="test_run",
        num_epochs_per_trial=10,
        num_samples=5,
    )
    assert config.perturbation_interval == 10
    assert config.time_attr == "training_iteration"


def test_read_from_dot_key_simple() -> None:
    """Test reading simple dot-separated key."""
    config = {"model": {"d_model": 64}}
    result = read_from_dot_key(config, "model.d_model")
    assert result == 64


def test_read_from_dot_key_nested() -> None:
    """Test reading deeply nested dot-separated key."""
    config = {"model": {"init_args": {"criterion": {"quantiles": [0.1, 0.5, 0.9]}}}}
    result = read_from_dot_key(config, "model.init_args.criterion.quantiles")
    assert result == [0.1, 0.5, 0.9]


def test_read_from_dot_key_single() -> None:
    """Test reading single key (no dots)."""
    config = {"batch_size": 32}
    result = read_from_dot_key(config, "batch_size")
    assert result == 32


def test_apply_key_string_path() -> None:
    """Test apply_key with string path."""
    config = {}
    apply_key(config, "model.d_model", 128)
    assert config == {"model": {"d_model": 128}}


def test_apply_key_list_path() -> None:
    """Test apply_key with list path."""
    config = {}
    apply_key(config, ["optimizer", "init_args", "lr"], 0.01)
    assert config == {"optimizer": {"init_args": {"lr": 0.01}}}


def test_apply_key_existing_structure() -> None:
    """Test apply_key modifies existing structure."""
    config = {"model": {"d_model": 64, "num_heads": 4}}
    apply_key(config, "model.num_layers", 3)
    assert config == {"model": {"d_model": 64, "num_heads": 4, "num_layers": 3}}


def test_apply_key_overwrites_existing() -> None:
    """Test apply_key overwrites existing values."""
    config = {"model": {"d_model": 64}}
    apply_key(config, "model.d_model", 128)
    assert config == {"model": {"d_model": 128}}


def test_apply_key_deep_nesting() -> None:
    """Test apply_key creates deep nested structure."""
    config = {}
    apply_key(config, "a.b.c.d.e", "deep_value")
    assert config == {"a": {"b": {"c": {"d": {"e": "deep_value"}}}}}


def test_apply_config_single_param() -> None:
    """Test apply_config with single parameter."""
    cli_config = {"model": {"init_args": {"d_model": 64}}}
    hyperparams = {"d_model": 128}
    param2key = {"d_model": "model.init_args.d_model"}

    result = apply_config(hyperparams, cli_config, param2key)

    assert result["model"]["init_args"]["d_model"] == 128
    # Original should not be mutated
    assert cli_config["model"]["init_args"]["d_model"] == 64


def test_apply_config_multiple_params() -> None:
    """Test apply_config with multiple parameters."""
    cli_config = {
        "model": {"init_args": {"d_model": 64, "num_heads": 4}},
        "optimizer": {"init_args": {"lr": 0.001}},
    }
    hyperparams = {"d_model": 128, "lr": 0.01, "batch_size": 32}
    param2key = {
        "d_model": "model.init_args.d_model",
        "lr": "optimizer.init_args.lr",
        "batch_size": "data.init_args.batch_size",
    }

    result = apply_config(hyperparams, cli_config, param2key)

    assert result["model"]["init_args"]["d_model"] == 128
    assert result["optimizer"]["init_args"]["lr"] == 0.01
    assert result["data"]["init_args"]["batch_size"] == 32


def test_apply_config_creates_new_paths() -> None:
    """Test apply_config creates new nested paths."""
    cli_config = {"trainer": {"max_epochs": 100}}
    hyperparams = {"new_param": "value"}
    param2key = {"new_param": "new.nested.path"}

    result = apply_config(hyperparams, cli_config, param2key)

    assert result["new"]["nested"]["path"] == "value"
    assert result["trainer"]["max_epochs"] == 100


def test_make_tune_callback() -> None:
    """Test make_tune_callback creates correct callback config."""
    metrics = ["loss", "accuracy", "rmse"]
    result = make_tune_callback(metrics)

    expected = {
        "class_path": "transformertf.utils.tune.TuneReportCallback",
        "init_args": {
            "on": "validation_end",
            "metrics": metrics,
            "save_checkpoints": False,
        },
    }
    assert result == expected


def test_make_tune_callback_empty_metrics() -> None:
    """Test make_tune_callback with empty metrics."""
    result = make_tune_callback([])

    assert result["init_args"]["metrics"] == []


def test_make_tune_callback_none_metrics() -> None:
    """Test make_tune_callback with None metrics."""
    result = make_tune_callback(None)

    assert result["init_args"]["metrics"] == []


def test_find_optimizer_params() -> None:
    """Test find_optimizer_params extracts optimizer parameters."""
    cli_config = {
        "optimizer": {
            "class_path": "torch.optim.SGD",
            "init_args": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
        }
    }
    result = find_optimizer_params(cli_config)

    expected = {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4}
    assert result == expected


def test_find_optimizer_params_missing_optimizer() -> None:
    """Test find_optimizer_params with missing optimizer section."""
    cli_config = {"model": {"class_path": "MyModel"}}
    result = find_optimizer_params(cli_config)

    assert result == {}


def test_find_optimizer_params_missing_init_args() -> None:
    """Test find_optimizer_params with missing init_args."""
    cli_config = {"optimizer": {"class_path": "torch.optim.SGD"}}
    result = find_optimizer_params(cli_config)

    assert result == {}


def test_find_lr_scheduler_params() -> None:
    """Test find_lr_scheduler_params extracts scheduler parameters."""
    cli_config = {
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 10, "gamma": 0.1},
        }
    }
    result = find_lr_scheduler_params(cli_config)

    expected = {"step_size": 10, "gamma": 0.1}
    assert result == expected


def test_find_lr_scheduler_params_missing() -> None:
    """Test find_lr_scheduler_params with missing scheduler."""
    cli_config = {"optimizer": {"class_path": "torch.optim.SGD"}}
    result = find_lr_scheduler_params(cli_config)

    assert result == {}


def test_make_set_optimizer_params_callback() -> None:
    """Test make_set_optimizer_params_callback creates correct callback."""
    cli_config = {
        "optimizer": {"init_args": {"lr": 0.01, "momentum": 0.9}},
        "lr_scheduler": {"init_args": {"step_size": 10, "gamma": 0.1}},
    }
    result = make_set_optimizer_params_callback(cli_config)

    expected = {
        "class_path": "transformertf.callbacks.SetOptimizerParamsCallback",
        "init_args": {
            "lr": 0.01,
            "momentum": 0.9,
            "step_size": 10,
            "gamma": 0.1,
        },
    }
    assert result == expected


def test_make_set_optimizer_params_callback_optimizer_only() -> None:
    """Test make_set_optimizer_params_callback with optimizer only."""
    cli_config = {"optimizer": {"init_args": {"lr": 0.01, "momentum": 0.9}}}
    result = make_set_optimizer_params_callback(cli_config)

    expected = {
        "class_path": "transformertf.callbacks.SetOptimizerParamsCallback",
        "init_args": {
            "lr": 0.01,
            "momentum": 0.9,
        },
    }
    assert result == expected


def test_load_cli_config() -> None:
    """Test load_cli_config loads and modifies YAML config."""
    config_data = {
        "trainer": {"max_epochs": 100, "enable_progress_bar": True},
        "model": {"class_path": "MyModel"},
        "ckpt_path": "checkpoint.ckpt",
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        tune_config = Mock()
        tune_config.cli_config_path = temp_path
        tune_config.num_epochs_per_trial = 50

        result = load_cli_config(tune_config)

        # Check modifications
        assert result["trainer"]["max_epochs"] == 50  # Overridden
        assert result["trainer"]["enable_progress_bar"] is False  # Set to False
        assert "ckpt_path" not in result  # Removed
        assert "plugins" in result["trainer"]  # Ray plugin added

        # Check original data preserved
        assert result["model"]["class_path"] == "MyModel"
    finally:
        Path(temp_path).unlink()


def test_load_cli_config_missing_file() -> None:
    """Test load_cli_config raises error for missing file."""
    tune_config = Mock()
    tune_config.cli_config_path = "nonexistent.yml"

    with pytest.raises(FileNotFoundError, match="File not found"):
        load_cli_config(tune_config)


def test_load_cli_config_non_yaml() -> None:
    """Test load_cli_config raises error for non-YAML file."""
    tune_config = Mock()
    tune_config.cli_config_path = "config.txt"

    with pytest.raises(ValueError, match="File must be a YAML file"):
        load_cli_config(tune_config)


@pytest.mark.parametrize("extension", [".yaml", ".yml"])
def test_load_cli_config_yaml_extensions(extension: str) -> None:
    """Test load_cli_config accepts both .yaml and .yml extensions."""
    config_data = {"trainer": {"max_epochs": 100}}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=extension, delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        tune_config = Mock()
        tune_config.cli_config_path = temp_path
        tune_config.num_epochs_per_trial = 50

        result = load_cli_config(tune_config)
        assert result["trainer"]["max_epochs"] == 50
    finally:
        Path(temp_path).unlink()


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_configure_asha_scheduler() -> None:
    """Test configure_asha_scheduler creates ASHA scheduler."""
    from transformertf.utils.tune import configure_asha_scheduler

    scheduler = configure_asha_scheduler(
        num_epochs=100,
        patience=10,
        reduction_factor=2,
    )

    # Basic type check since we can't easily inspect Ray Tune objects
    assert scheduler is not None
    assert "Scheduler" in str(type(scheduler))


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_configure_pbt_scheduler() -> None:
    """Test configure_pbt_scheduler creates PBT scheduler."""
    from transformertf.utils.tune import configure_pbt_scheduler

    grid = {"lr": Mock()}
    scheduler = configure_pbt_scheduler(
        monitor="loss",
        grid=grid,
        perturbation_interval=20,
        time_attr="epoch",
    )

    # Basic type check since we can't easily inspect Ray Tune objects
    assert scheduler is not None
    assert "PopulationBasedTraining" in str(type(scheduler))


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_configure_search_alg() -> None:
    """Test configure_search_alg creates HyperOpt search algorithm."""
    from transformertf.utils.tune import configure_search_alg

    search_alg = configure_search_alg(monitor="accuracy")

    # Basic type check since we can't easily inspect Ray Tune objects
    assert search_alg is not None


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_configure_reporter() -> None:
    """Test configure_reporter creates CLI reporter."""
    from transformertf.utils.tune import configure_reporter

    grid = {"lr": Mock(), "batch_size": Mock()}
    metrics = ["loss", "accuracy"]

    reporter = configure_reporter(grid, metrics)

    # Basic type check since we can't easily inspect Ray Tune objects
    assert reporter is not None


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_tune_unified_interface() -> None:
    """Test the unified tune interface with mocking."""
    from transformertf.utils.tune import tune

    # Create a minimal config file
    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.lstm.LSTM",
                "init_args": {"d_model": 64},
            },
            "data": {"class_path": "transformertf.data.TimeSeriesDataModule"},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "logging_metrics": ["RMSE/val"],
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Mock Ray Tune components to avoid actual training
        with patch("ray.tune.Tuner") as mock_tuner_class:
            mock_tuner = Mock()
            mock_results = Mock()
            mock_best_result = Mock()
            mock_best_result.metrics = {"loss/val": 0.5}
            mock_best_result.config = {"model.init_args.d_model": 64}
            mock_results.get_best_result.return_value = mock_best_result
            mock_tuner.fit.return_value = mock_results
            mock_tuner_class.return_value = mock_tuner

            # Mock the LightningCLI to avoid actual training
            with patch("transformertf.main.LightningCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli.trainer.fit = Mock()
                mock_cli_class.return_value = mock_cli

                # Run the tune function
                results = tune(temp_path)

                # Verify mocks were called
                assert mock_tuner_class.called
                assert mock_tuner.fit.called
                assert results == mock_results
    finally:
        Path(temp_path).unlink()


def test_apply_config_preserves_types() -> None:
    """Test apply_config preserves various data types."""
    cli_config = {"existing": {"value": "old"}}
    hyperparams = {
        "int_val": 42,
        "float_val": 3.14,
        "str_val": "hello",
        "bool_val": True,
        "list_val": [1, 2, 3],
        "dict_val": {"nested": "value"},
    }
    param2key = {
        "int_val": "data.int_val",
        "float_val": "data.float_val",
        "str_val": "data.str_val",
        "bool_val": "data.bool_val",
        "list_val": "data.list_val",
        "dict_val": "data.dict_val",
    }

    result = apply_config(hyperparams, cli_config, param2key)

    assert result["data"]["int_val"] == 42
    assert result["data"]["float_val"] == 3.14
    assert result["data"]["str_val"] == "hello"
    assert result["data"]["bool_val"] is True
    assert result["data"]["list_val"] == [1, 2, 3]
    assert result["data"]["dict_val"] == {"nested": "value"}


def test_read_from_dot_key_error_handling() -> None:
    """Test read_from_dot_key handles missing keys gracefully."""
    config = {"model": {"d_model": 64}}

    with pytest.raises(KeyError):
        read_from_dot_key(config, "model.missing_key")

    with pytest.raises(KeyError):
        read_from_dot_key(config, "missing_section.d_model")


# Tests for new search algorithms


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_ax_search_algorithm_configuration() -> None:
    """Test Ax search algorithm configuration."""
    import tempfile

    import yaml

    from transformertf.utils.tune import tune

    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.lstm.LSTM",
                "init_args": {"d_model": 64},
            },
            "data": {"class_path": "transformertf.data.TimeSeriesDataModule"},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "search_algorithm": {
                "type": "ax",
                "num_bootstrap": 20,
                "min_trials_observed": 5,
                "verbose_logging": False,
            },
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Should handle missing Ax dependency gracefully
        with pytest.raises(ImportError, match="Ax is required"):
            tune(temp_path)
    finally:
        Path(temp_path).unlink()


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_bayesopt_search_algorithm_configuration() -> None:
    """Test BayesOpt search algorithm configuration."""
    import tempfile

    import yaml

    from transformertf.utils.tune import tune

    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.lstm.LSTM",
                "init_args": {"d_model": 64},
            },
            "data": {"class_path": "transformertf.data.TimeSeriesDataModule"},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "search_algorithm": {
                "type": "bayesopt",
                "random_state": 42,
                "random_search_steps": 10,
                "utility_kwargs": {"kind": "ucb", "kappa": 2.576},
            },
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Should handle missing BayesOpt dependency gracefully
        with pytest.raises(ImportError, match="BayesOpt is required"):
            tune(temp_path)
    finally:
        Path(temp_path).unlink()


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_bohb_search_algorithm_configuration() -> None:
    """Test BOHB search algorithm configuration."""
    import tempfile

    import yaml

    from transformertf.utils.tune import tune

    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.lstm.LSTM",
                "init_args": {"d_model": 64},
            },
            "data": {"class_path": "transformertf.data.TimeSeriesDataModule"},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "search_algorithm": {
                "type": "bohb",
            },
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Should handle missing BOHB dependency gracefully
        with pytest.raises(ImportError, match="BOHB is required"):
            tune(temp_path)
    finally:
        Path(temp_path).unlink()


@pytest.mark.skipif(
    not pytest.importorskip("ray.tune", reason="Ray Tune not available"),
    reason="Ray Tune not available",
)
def test_nevergrad_search_algorithm_configuration() -> None:
    """Test Nevergrad search algorithm configuration."""
    import tempfile

    import yaml

    from transformertf.utils.tune import tune

    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.lstm.LSTM",
                "init_args": {"d_model": 64},
            },
            "data": {"class_path": "transformertf.data.TimeSeriesDataModule"},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "search_algorithm": {
                "type": "nevergrad",
                "optimizer": "NGOpt",
            },
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Should handle missing Nevergrad dependency gracefully
        with pytest.raises(ImportError, match="Nevergrad is required"):
            tune(temp_path)
    finally:
        Path(temp_path).unlink()


def test_unknown_search_algorithm() -> None:
    """Test handling of unknown search algorithm."""
    import tempfile

    import yaml

    from transformertf.utils.tune import tune

    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.lstm.LSTM",
                "init_args": {"d_model": 64},
            },
            "data": {"class_path": "transformertf.data.TimeSeriesDataModule"},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64, 128]}
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "search_algorithm": {"type": "unknown_algorithm"},
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Should run without error (unknown algorithm defaults to None)
        with patch("ray.tune.Tuner") as mock_tuner_class:
            mock_tuner = Mock()
            mock_results = Mock()
            mock_best_result = Mock()
            mock_best_result.metrics = {"loss/val": 0.5}
            mock_best_result.config = {"model.init_args.d_model": 64}
            mock_results.get_best_result.return_value = mock_best_result
            mock_tuner.fit.return_value = mock_results
            mock_tuner_class.return_value = mock_tuner

            with patch("transformertf.main.LightningCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli.trainer.fit = Mock()
                mock_cli_class.return_value = mock_cli

                results = tune(temp_path)
                assert results == mock_results
    finally:
        Path(temp_path).unlink()
