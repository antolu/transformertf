from __future__ import annotations

import typing
import warnings

import pytest
import torch

from transformertf.models.temporal_conv_transformer import (
    TemporalConvTransformer,
    TemporalConvTransformerModel,
)


@pytest.fixture(scope="module")
def tct_config() -> dict[str, typing.Any]:
    """Standard configuration for TCT testing."""
    return {
        "num_past_features": 8,
        "num_future_features": 4,
        "output_dim": 1,
        "d_hidden": 64,
        "num_attention_heads": 4,
        "compression_factor": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dropout": 0.1,
        "max_dilation": 8,
    }


@pytest.fixture(scope="module")
def small_tct_config() -> dict[str, typing.Any]:
    """Small configuration for faster testing."""
    return {
        "num_past_features": 4,
        "num_future_features": 2,
        "output_dim": 1,
        "d_hidden": 32,
        "num_attention_heads": 2,
        "compression_factor": 2,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dropout": 0.0,  # No dropout for deterministic tests
        "max_dilation": 4,
    }


@pytest.fixture(scope="module")
def tct_lightning_module(tct_config: dict[str, typing.Any]) -> TemporalConvTransformer:
    """Create TCT Lightning module for testing."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", UserWarning
        )  # Suppress sequence length warnings
        return TemporalConvTransformer(**tct_config)


@pytest.fixture(scope="module")
def small_tct_lightning_module(
    small_tct_config: dict[str, typing.Any],
) -> TemporalConvTransformer:
    """Create small TCT Lightning module for testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return TemporalConvTransformer(**small_tct_config)


@pytest.fixture(scope="module")
def tct_core_model(tct_config: dict[str, typing.Any]) -> TemporalConvTransformerModel:
    """Create TCT core model for testing."""
    return TemporalConvTransformerModel(**tct_config)


@pytest.fixture(scope="module")
def small_tct_core_model(
    small_tct_config: dict[str, typing.Any],
) -> TemporalConvTransformerModel:
    """Create small TCT core model for testing."""
    return TemporalConvTransformerModel(**small_tct_config)


@pytest.fixture(scope="module")
def adequate_batch() -> dict[str, torch.Tensor]:
    """Batch with adequate sequence lengths for compression_factor=4."""
    return {
        "encoder_input": torch.randn(4, 400, 8),  # Above 384 minimum
        "decoder_input": torch.randn(4, 100, 4),  # Above 32 minimum
        "encoder_lengths": torch.full((4, 1), 400),
        "decoder_lengths": torch.full((4, 1), 100),
        "target": torch.randn(4, 100, 1),
    }


@pytest.fixture(scope="module")
def small_adequate_batch() -> dict[str, torch.Tensor]:
    """Batch with adequate sequence lengths for compression_factor=2."""
    return {
        "encoder_input": torch.randn(2, 200, 4),  # Above 96 minimum
        "decoder_input": torch.randn(2, 50, 2),  # Above 16 minimum
        "encoder_lengths": torch.full((2, 1), 200),
        "decoder_lengths": torch.full((2, 1), 50),
        "target": torch.randn(2, 50, 1),
    }


@pytest.fixture(scope="module")
def short_batch() -> dict[str, torch.Tensor]:
    """Batch with short sequences that will trigger warnings."""
    return {
        "encoder_input": torch.randn(2, 50, 8),  # Too short for compression_factor=4
        "decoder_input": torch.randn(2, 20, 4),  # Too short
        "encoder_lengths": torch.full((2, 1), 50),
        "decoder_lengths": torch.full((2, 1), 20),
        "target": torch.randn(2, 20, 1),
    }


@pytest.fixture(scope="module")
def predict_batch() -> dict[str, torch.Tensor]:
    """Batch for prediction testing (no target)."""
    return {
        "encoder_input": torch.randn(2, 400, 8),
        "decoder_input": torch.randn(2, 100, 4),
        "encoder_lengths": torch.full((2, 1), 400),
        "decoder_lengths": torch.full((2, 1), 100),
    }


@pytest.fixture(scope="module")
def large_batch() -> dict[str, torch.Tensor]:
    """Large batch for memory efficiency testing."""
    return {
        "encoder_input": torch.randn(8, 1000, 16),
        "decoder_input": torch.randn(8, 250, 8),
        "encoder_lengths": torch.full((8, 1), 1000),
        "decoder_lengths": torch.full((8, 1), 250),
        "target": torch.randn(8, 250, 1),
    }


@pytest.fixture
def deterministic_batch() -> dict[str, torch.Tensor]:
    """Deterministic batch for reproducibility testing."""
    torch.manual_seed(42)
    return {
        "encoder_input": torch.randn(2, 200, 6),
        "decoder_input": torch.randn(2, 50, 3),
        "target": torch.randn(2, 50, 1),
    }


@pytest.fixture(scope="module", params=[2, 4, 8])
def compression_factor(request) -> int:
    """Parametrized compression factors for testing."""
    return request.param


@pytest.fixture(scope="module", params=[1, 3, 5, 7])
def output_dim(request) -> int:
    """Parametrized output dimensions for testing."""
    return request.param


@pytest.fixture(scope="module", params=[32, 64, 128])
def hidden_dim(request) -> int:
    """Parametrized hidden dimensions for testing."""
    return request.param


@pytest.fixture
def tct_config_factory():
    """Factory for creating custom TCT configurations."""

    def _create_config(**overrides):
        base_config = {
            "num_past_features": 8,
            "num_future_features": 4,
            "output_dim": 1,
            "d_hidden": 64,
            "num_attention_heads": 4,
            "compression_factor": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dropout": 0.1,
            "max_dilation": 8,
        }
        base_config.update(overrides)
        return base_config

    return _create_config


@pytest.fixture
def batch_factory():
    """Factory for creating custom batches."""

    def _create_batch(
        batch_size: int = 2,
        encoder_len: int = 400,
        decoder_len: int = 100,
        num_past_features: int = 8,
        num_future_features: int = 4,
        include_target: bool = True,
        device: str = "cpu",
    ):
        batch = {
            "encoder_input": torch.randn(
                batch_size, encoder_len, num_past_features, device=device
            ),
            "decoder_input": torch.randn(
                batch_size, decoder_len, num_future_features, device=device
            ),
            "encoder_lengths": torch.full((batch_size, 1), encoder_len, device=device),
            "decoder_lengths": torch.full((batch_size, 1), decoder_len, device=device),
        }

        if include_target:
            batch["target"] = torch.randn(batch_size, decoder_len, 1, device=device)

        return batch

    return _create_batch


@pytest.fixture
def model_factory():
    """Factory for creating TCT models with custom configurations."""

    def _create_model(
        model_type: str = "lightning",  # "lightning" or "core"
        suppress_warnings: bool = True,
        **config_overrides,
    ):
        base_config = {
            "num_past_features": 8,
            "num_future_features": 4,
            "output_dim": 1,
            "d_hidden": 32,  # Smaller for testing
            "compression_factor": 2,  # Lower for testing
            "dropout": 0.0,  # Deterministic
        }
        base_config.update(config_overrides)

        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if model_type == "lightning":
                    return TemporalConvTransformer(**base_config)
                return TemporalConvTransformerModel(**base_config)
        else:
            if model_type == "lightning":
                return TemporalConvTransformer(**base_config)
            return TemporalConvTransformerModel(**base_config)

    return _create_model


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cuda_batch(batch_factory, device):
    """Create batch on CUDA device if available."""
    if device.type == "cuda":
        return batch_factory(device="cuda")
    pytest.skip("CUDA not available")


@pytest.fixture
def seed_everything():
    """Set random seeds for reproducible testing."""

    def _seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

    return _seed
