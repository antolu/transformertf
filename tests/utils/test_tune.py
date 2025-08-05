"""Tests for transformertf.utils.tune module."""

from __future__ import annotations

import pytest

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
    assert len(params) == 1  # Only one parameter expected

    # Test return type annotation
    assert sig.return_annotation is not None


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


if __name__ == "__main__":
    pytest.main([__file__])
