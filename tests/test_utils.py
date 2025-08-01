"""Utility functions for testing."""

from __future__ import annotations

import typing
from contextlib import contextmanager
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import LightningModule
from torch.utils.data import DataLoader

from transformertf.data import DataModuleBase


def assert_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    msg: str = "",
) -> None:
    """Assert that a tensor has the expected shape."""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}. {msg}"
    )


def assert_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    msg: str = "",
) -> None:
    """Assert that a tensor has the expected dtype."""
    assert tensor.dtype == expected_dtype, (
        f"Expected dtype {expected_dtype}, got {tensor.dtype}. {msg}"
    )


def assert_tensor_finite(tensor: torch.Tensor, msg: str = "") -> None:
    """Assert that all values in a tensor are finite."""
    assert torch.isfinite(tensor).all(), f"Tensor contains non-finite values. {msg}"


def assert_tensor_not_nan(tensor: torch.Tensor, msg: str = "") -> None:
    """Assert that tensor contains no NaN values."""
    assert not torch.isnan(tensor).any(), f"Tensor contains NaN values. {msg}"


def assert_tensor_range(
    tensor: torch.Tensor,
    min_val: float | None = None,
    max_val: float | None = None,
    msg: str = "",
) -> None:
    """Assert that tensor values are within expected range."""
    if min_val is not None:
        assert tensor.min() >= min_val, (
            f"Tensor minimum {tensor.min()} < {min_val}. {msg}"
        )
    if max_val is not None:
        assert tensor.max() <= max_val, (
            f"Tensor maximum {tensor.max()} > {max_val}. {msg}"
        )


def assert_model_parameters_updated(
    model: LightningModule,
    initial_params: dict[str, torch.Tensor],
    tolerance: float = 1e-6,
) -> None:
    """Assert that model parameters have been updated during training."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_param = initial_params[name]
            assert not torch.allclose(param, initial_param, atol=tolerance), (
                f"Parameter {name} was not updated during training"
            )


def assert_dataloader_consistency(dataloader: DataLoader) -> None:
    """Assert that dataloader produces consistent batches."""
    batch_sizes = []
    for batch in dataloader:
        if isinstance(batch, dict):
            # Get batch size from first tensor in batch
            first_tensor = next(iter(batch.values()))
            batch_sizes.append(first_tensor.shape[0])
        else:
            batch_sizes.append(batch.shape[0])

    # All batches should have same size except possibly the last one
    if len(batch_sizes) > 1:
        assert all(size == batch_sizes[0] for size in batch_sizes[:-1]), (
            "Inconsistent batch sizes in dataloader"
        )


def assert_gradient_flow(model: LightningModule) -> None:
    """Assert that gradients flow properly through the model."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert_tensor_finite(param.grad, f"Gradient for {name} is not finite")


def create_mock_datamodule(
    train_size: int = 100,
    val_size: int = 50,
    batch_size: int = 16,
    **kwargs: typing.Any,
) -> Mock:
    """Create a mock DataModule for testing."""
    mock_dm = Mock(spec=DataModuleBase)
    mock_dm.hparams = {"batch_size": batch_size, **kwargs}

    # Mock datasets
    mock_train_dataset = Mock()
    mock_train_dataset.__len__ = Mock(return_value=train_size)
    mock_val_dataset = Mock()
    mock_val_dataset.__len__ = Mock(return_value=val_size)

    mock_dm.train_dataset = mock_train_dataset
    mock_dm.val_dataset = mock_val_dataset

    return mock_dm


@contextmanager
def temporary_seed(seed: int = 42):
    """Context manager for temporarily setting random seeds."""
    # Store original states
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        yield
    finally:
        # Restore original states
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)


def check_model_device(model: LightningModule, expected_device: str) -> None:
    """Check that model is on the expected device."""
    for param in model.parameters():
        assert str(param.device) == expected_device, (
            f"Parameter on device {param.device}, expected {expected_device}"
        )


def validate_dataframe_structure(
    df: pd.DataFrame,
    required_columns: list[str],
    min_length: int = 1,
) -> None:
    """Validate that DataFrame has the expected structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_length, f"DataFrame too short: {len(df)} < {min_length}"

    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
        assert not df[col].isna().all(), f"Column {col} is all NaN"


def approximate_equal(
    a: float | torch.Tensor,
    b: float | torch.Tensor,
    tolerance: float = 1e-6,
) -> bool:
    """Check if two values are approximately equal."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a, b, atol=tolerance)
    return abs(a - b) < tolerance


def create_time_series(
    length: int = 1000,
    num_features: int = 3,
    trend: bool = True,
    seasonal: bool = True,
    noise_std: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic time series data."""
    with temporary_seed(seed):
        time = np.linspace(0, 100, length)
        data = {"time_ms": time * 1000}

        for i in range(num_features):
            values = np.zeros(length)

            if trend:
                values += 0.1 * time + np.random.normal(0, 0.05)

            if seasonal:
                # Multiple seasonal components
                values += 2 * np.sin(2 * np.pi * time / 10)
                values += 0.5 * np.sin(2 * np.pi * time / 3)

            if noise_std > 0:
                values += np.random.normal(0, noise_std, length)

            data[f"feature_{i}"] = values

        return pd.DataFrame(data)


def create_hysteresis_data(
    length: int = 1000,
    amplitude: float = 1.0,
    frequency: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create hysteresis-like data for physics applications."""
    with temporary_seed(seed):
        time = np.linspace(0, 100, length)

        # Create input signal (current)
        current = amplitude * np.sin(2 * np.pi * frequency * time)

        # Create hysteresis response (magnetic field)
        field = np.zeros(length)
        for i in range(1, length):
            # Simple hysteresis model
            field[i] = (
                0.8 * field[i - 1]
                + 0.2 * current[i]
                + 0.1 * np.sign(current[i] - current[i - 1])
            )

        # Add noise
        current += np.random.normal(0, 0.01, length)
        field += np.random.normal(0, 0.01, length)

        return pd.DataFrame({
            "time_ms": time * 1000,
            "I_meas_A": current,
            "B_meas_T": field,
        })


# Pytest fixtures for common test scenarios
@pytest.fixture
def sample_time_series():
    """Fixture providing sample time series data."""
    return create_time_series()


@pytest.fixture
def sample_hysteresis_data():
    """Fixture providing sample hysteresis data."""
    return create_hysteresis_data()


@pytest.fixture
def device():
    """Fixture providing the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def deterministic_environment():
    """Fixture for deterministic test environment."""
    with temporary_seed(42):
        yield
