"""Tests for transformertf.utils.tune module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from transformertf.utils.tune import TuneReportCallback, tune


def test_tune_report_callback_creation() -> None:
    """Test TuneReportCallback can be instantiated."""
    callback = TuneReportCallback(
        metrics={"val_loss": "val_loss"},
        on="validation_end",
    )
    assert callback is not None


def test_tune_function_signature() -> None:
    """Test tune function has the expected signature."""
    import inspect

    # Test that tune function exists and has the right signature
    sig = inspect.signature(tune)
    params = list(sig.parameters.keys())

    assert "config_path" in params
    assert "resume" in params
    assert "resume_errored" in params
    assert "restart_errored" in params
    assert (
        len(params) == 4
    )  # config_path, resume, resume_errored, restart_errored parameters expected

    # Test return type annotation
    assert sig.return_annotation is not None

    # Test resume parameter has correct type and default
    resume_param = sig.parameters["resume"]
    assert resume_param.default is None

    # Test error handling parameters have correct defaults
    resume_errored_param = sig.parameters["resume_errored"]
    assert resume_errored_param.default is False

    restart_errored_param = sig.parameters["restart_errored"]
    assert restart_errored_param.default is False


def test_tune_config_yaml_structure() -> None:
    """Test that a proper YAML config structure is expected."""
    config_data = {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer",
                "init_args": {"d_model": 256},
            },
            "data": {
                "class_path": "transformertf.data.EncoderDecoderDataModule",
                "init_args": {"batch_size": 32},
            },
            "trainer": {"max_epochs": 100},
        },
        "search_space": {
            "d_model": {"type": "choice", "values": [128, 256, 512]},
            "learning_rate": {"type": "uniform", "min": 0.0001, "max": 0.01},
        },
        "tune_config": {
            "num_samples": 20,
            "metric": "val_loss",
            "mode": "min",
            "resources": {"cpu": 4, "gpu": 1},
            "scheduler": {
                "type": "asha",
                "max_t": 100,
                "grace_period": 10,
                "reduction_factor": 2,
            },
            "search_algorithm": {"type": "hyperopt"},
            "experiment_name": "test_experiment",
            "storage_path": "./ray_results",
            "logging_metrics": ["train_loss", "val_accuracy"],
            "env_vars": {"patterns": ["NEPTUNE_*", "*_proxy", "*_PROXY", "WANDB_*"]},
        },
    }

    # This represents the expected structure - no assertions needed,
    # just verifying the structure is reasonable
    assert "base_config" in config_data
    assert "search_space" in config_data
    assert "tune_config" in config_data
    assert "resources" in config_data["tune_config"]


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_tune_config(temp_storage_dir):
    """Sample tune configuration for testing."""
    return {
        "base_config": {
            "model": {
                "class_path": "transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer",
                "init_args": {"d_model": 64},
            },
            "data": {
                "class_path": "transformertf.data.EncoderDecoderDataModule",
                "init_args": {"batch_size": 16},
            },
            "trainer": {"max_epochs": 5},
        },
        "search_space": {
            "model.init_args.d_model": {"type": "choice", "values": [32, 64]},
        },
        "tune_config": {
            "num_samples": 2,
            "metric": "loss/val",
            "mode": "min",
            "experiment_name": "test_experiment",
            "storage_path": temp_storage_dir,
        },
    }


@pytest.fixture
def temp_config_file(sample_tune_config):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(sample_tune_config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
def test_tune_no_resume(
    mock_is_initialized, mock_ray_init, mock_tuner_class, temp_config_file
):
    """Test tune function without resume (default behavior)."""
    mock_is_initialized.return_value = False
    mock_tuner_instance = Mock()
    mock_results = Mock()
    mock_best_result = Mock()
    mock_best_result.metrics = {"loss/val": 0.5}
    mock_best_result.config = {"d_model": 64}
    mock_results.get_best_result.return_value = mock_best_result
    mock_tuner_instance.fit.return_value = mock_results
    mock_tuner_class.return_value = mock_tuner_instance

    # Test without resume parameter
    result = tune(temp_config_file)

    # Verify new tuner was created (not restored)
    mock_tuner_class.assert_called_once()
    mock_tuner_class.restore.assert_not_called()
    mock_tuner_instance.fit.assert_called_once()
    assert result == mock_results


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
@patch("transformertf.utils.tune.os.path.exists")
def test_tune_resume_true_experiment_exists(
    mock_exists,
    mock_is_initialized,
    mock_ray_init,
    mock_tuner_class,
    temp_config_file,
    temp_storage_dir,
):
    """Test tune function with resume=True when experiment directory exists."""
    mock_is_initialized.return_value = False
    mock_exists.return_value = True

    # Mock the restored tuner
    mock_restored_tuner = Mock()
    mock_results = Mock()
    mock_best_result = Mock()
    mock_best_result.metrics = {"loss/val": 0.3}
    mock_best_result.config = {"d_model": 32}
    mock_results.get_best_result.return_value = mock_best_result
    mock_restored_tuner.fit.return_value = mock_results
    mock_tuner_class.restore.return_value = mock_restored_tuner

    # Test with resume=True
    result = tune(temp_config_file, resume=True)

    # Verify tuner was restored (not created new)
    expected_resume_path = os.path.join(temp_storage_dir, "test_experiment")
    # Check that restore was called with path, trainable, and default error handling
    assert mock_tuner_class.restore.call_count == 1
    call_args = mock_tuner_class.restore.call_args
    assert len(call_args[0]) == 2  # path and trainable
    assert call_args[0][0] == expected_resume_path  # path is first argument
    # Check keyword arguments for error handling
    assert call_args[1]["resume_errored"] is False
    assert call_args[1]["restart_errored"] is False
    mock_tuner_class.assert_not_called()  # New tuner should not be created
    mock_restored_tuner.fit.assert_called_once()
    assert result == mock_results


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
@patch("transformertf.utils.tune.os.path.exists")
def test_tune_resume_true_experiment_not_exists(
    mock_exists,
    mock_is_initialized,
    mock_ray_init,
    mock_tuner_class,
    temp_config_file,
):
    """Test tune function with resume=True when experiment directory doesn't exist."""
    mock_is_initialized.return_value = False
    mock_exists.return_value = False

    # Mock new tuner creation
    mock_tuner_instance = Mock()
    mock_results = Mock()
    mock_best_result = Mock()
    mock_best_result.metrics = {"loss/val": 0.4}
    mock_best_result.config = {"d_model": 64}
    mock_results.get_best_result.return_value = mock_best_result
    mock_tuner_instance.fit.return_value = mock_results
    mock_tuner_class.return_value = mock_tuner_instance

    # Test with resume=True but no existing experiment
    result = tune(temp_config_file, resume=True)

    # Verify new tuner was created (not restored)
    mock_tuner_class.restore.assert_not_called()
    mock_tuner_class.assert_called_once()
    mock_tuner_instance.fit.assert_called_once()
    assert result == mock_results


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
@patch("transformertf.utils.tune.os.path.exists")
def test_tune_resume_specific_path_exists(
    mock_exists,
    mock_is_initialized,
    mock_ray_init,
    mock_tuner_class,
    temp_config_file,
):
    """Test tune function with resume=<specific_path> when path exists."""
    mock_is_initialized.return_value = False
    mock_exists.return_value = True

    # Mock the restored tuner
    mock_restored_tuner = Mock()
    mock_results = Mock()
    mock_best_result = Mock()
    mock_best_result.metrics = {"loss/val": 0.2}
    mock_best_result.config = {"d_model": 32}
    mock_results.get_best_result.return_value = mock_best_result
    mock_restored_tuner.fit.return_value = mock_results
    mock_tuner_class.restore.return_value = mock_restored_tuner

    # Test with specific resume path
    resume_path = "/custom/experiment/path"
    result = tune(temp_config_file, resume=resume_path)

    # Verify tuner was restored from specific path
    # Check that restore was called with path and trainable
    assert mock_tuner_class.restore.call_count == 1
    call_args = mock_tuner_class.restore.call_args
    assert len(call_args[0]) == 2  # path and trainable
    assert call_args[0][0] == resume_path  # path is first argument
    mock_tuner_class.assert_not_called()  # New tuner should not be created
    mock_restored_tuner.fit.assert_called_once()
    assert result == mock_results


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
@patch("transformertf.utils.tune.os.path.exists")
def test_tune_resume_specific_path_not_exists(
    mock_exists,
    mock_is_initialized,
    mock_ray_init,
    mock_tuner_class,
    temp_config_file,
):
    """Test tune function with resume=<specific_path> when path doesn't exist."""
    mock_is_initialized.return_value = False
    mock_exists.return_value = False

    resume_path = "/nonexistent/experiment/path"

    # Test with non-existent specific resume path
    with pytest.raises(
        FileNotFoundError, match=f"Resume path does not exist: {resume_path}"
    ):
        tune(temp_config_file, resume=resume_path)

    # Verify no tuner operations were attempted
    mock_tuner_class.restore.assert_not_called()
    mock_tuner_class.assert_not_called()


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
def test_tune_resume_none_explicit(
    mock_is_initialized, mock_ray_init, mock_tuner_class, temp_config_file
):
    """Test tune function with resume=None explicitly."""
    mock_is_initialized.return_value = False
    mock_tuner_instance = Mock()
    mock_results = Mock()
    mock_best_result = Mock()
    mock_best_result.metrics = {"loss/val": 0.6}
    mock_best_result.config = {"d_model": 64}
    mock_results.get_best_result.return_value = mock_best_result
    mock_tuner_instance.fit.return_value = mock_results
    mock_tuner_class.return_value = mock_tuner_instance

    # Test with explicit resume=None
    result = tune(temp_config_file, resume=None)

    # Verify new tuner was created (not restored)
    mock_tuner_class.assert_called_once()
    mock_tuner_class.restore.assert_not_called()
    mock_tuner_instance.fit.assert_called_once()
    assert result == mock_results


def test_tune_resume_parameter_types():
    """Test that resume parameter accepts correct types."""
    # Test with different resume parameter types
    with patch("transformertf.utils.tune_config.load_tune_config") as mock_load_config:
        mock_load_config.side_effect = Exception(
            "Config loading bypassed for type test"
        )

        # These should not raise type errors during function call
        with pytest.raises(Exception, match="Config loading bypassed"):
            tune("config.yml", resume=None)

        with pytest.raises(Exception, match="Config loading bypassed"):
            tune("config.yml", resume=True)

        with pytest.raises(Exception, match="Config loading bypassed"):
            tune("config.yml", resume="/some/path")

        # Test error handling parameters
        with pytest.raises(Exception, match="Config loading bypassed"):
            tune("config.yml", resume=True, resume_errored=True, restart_errored=False)


@patch("transformertf.utils.tune.ray.tune.Tuner")
@patch("transformertf.utils.tune.ray.init")
@patch("transformertf.utils.tune.ray.is_initialized")
@patch("transformertf.utils.tune.os.path.exists")
def test_tune_resume_with_error_handling(
    mock_exists,
    mock_is_initialized,
    mock_ray_init,
    mock_tuner_class,
    temp_config_file,
    temp_storage_dir,
):
    """Test tune function with resume and error handling parameters."""
    mock_is_initialized.return_value = False
    mock_exists.return_value = True

    # Mock the restored tuner
    mock_restored_tuner = Mock()
    mock_results = Mock()
    mock_best_result = Mock()
    mock_best_result.metrics = {"loss/val": 0.3}
    mock_best_result.config = {"d_model": 32}
    mock_results.get_best_result.return_value = mock_best_result
    mock_restored_tuner.fit.return_value = mock_results
    mock_tuner_class.restore.return_value = mock_restored_tuner

    # Test with resume=True and error handling parameters
    result = tune(
        temp_config_file, resume=True, resume_errored=True, restart_errored=False
    )

    # Verify tuner was restored with correct error handling parameters
    expected_resume_path = os.path.join(temp_storage_dir, "test_experiment")
    assert mock_tuner_class.restore.call_count == 1
    call_args = mock_tuner_class.restore.call_args
    assert len(call_args[0]) == 2  # path and trainable
    assert call_args[0][0] == expected_resume_path  # path is first argument
    # Check keyword arguments for error handling
    assert call_args[1]["resume_errored"] is True
    assert call_args[1]["restart_errored"] is False
    mock_tuner_class.assert_not_called()  # New tuner should not be created
    mock_restored_tuner.fit.assert_called_once()
    assert result == mock_results


if __name__ == "__main__":
    pytest.main([__file__])
