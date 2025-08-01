"""Factory classes for creating test objects using factory_boy."""

from __future__ import annotations

import typing
from pathlib import Path

import factory
import numpy as np
import pandas as pd
import torch
from factory import fuzzy

from transformertf.data import (
    EncoderDecoderDataModule,
    TimeSeriesDataModule,
)
from transformertf.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
    TemporalFusionTransformerModel,
)


class TensorFactory(factory.Factory):
    """Factory for creating torch tensors."""

    class Meta:
        model = torch.Tensor

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        shape = kwargs.pop("shape", (10, 5))
        dtype = kwargs.pop("dtype", torch.float32)
        return torch.randn(shape, dtype=dtype)


class TimeSeriesDataFactory(factory.Factory):
    """Factory for creating time series data."""

    class Meta:
        model = pd.DataFrame

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        n_samples = kwargs.pop("n_samples", 1000)
        n_features = kwargs.pop("n_features", 3)

        # Create synthetic time series data
        time = np.linspace(0, 100, n_samples)
        data = {}

        for i in range(n_features):
            # Create correlated time series with trend and noise
            trend = 0.1 * time
            seasonal = 2 * np.sin(2 * np.pi * time / 10)
            noise = np.random.normal(0, 0.5, n_samples)
            data[f"feature_{i}"] = trend + seasonal + noise

        data["time_ms"] = time * 1000  # Convert to milliseconds

        return pd.DataFrame(data)


class ConfigFactory(factory.DictFactory):
    """Factory for creating configuration dictionaries."""

    ctxt_seq_len = fuzzy.FuzzyInteger(50, 200)
    tgt_seq_len = fuzzy.FuzzyInteger(10, 50)
    batch_size = fuzzy.FuzzyChoice([16, 32, 64])
    num_workers = 0
    learning_rate = fuzzy.FuzzyFloat(1e-4, 1e-2)


class TimeSeriesDataModuleFactory(factory.Factory):
    """Factory for creating TimeSeriesDataModule instances."""

    class Meta:
        model = TimeSeriesDataModule

    known_covariates = factory.LazyFunction(lambda: ["feature_0", "feature_1"])
    target_covariate = "feature_2"
    seq_len = fuzzy.FuzzyInteger(100, 300)
    batch_size = fuzzy.FuzzyChoice([16, 32, 64])
    num_workers = 0
    normalize = True
    stride = 1
    downsample = 1
    downsample_method = "interval"


class EncoderDecoderDataModuleFactory(factory.Factory):
    """Factory for creating EncoderDecoderDataModule instances."""

    class Meta:
        model = EncoderDecoderDataModule

    known_covariates = factory.LazyFunction(lambda: ["feature_0", "feature_1"])
    target_covariate = "feature_2"
    ctxt_seq_len = fuzzy.FuzzyInteger(50, 200)
    tgt_seq_len = fuzzy.FuzzyInteger(10, 50)
    batch_size = fuzzy.FuzzyChoice([16, 32, 64])
    num_workers = 0


class TFTModelFactory(factory.Factory):
    """Factory for creating TemporalFusionTransformerModel instances."""

    class Meta:
        model = TemporalFusionTransformerModel

    num_past_features = fuzzy.FuzzyInteger(2, 5)
    num_future_features = fuzzy.FuzzyInteger(1, 3)
    ctxt_seq_len = fuzzy.FuzzyInteger(50, 200)
    tgt_seq_len = fuzzy.FuzzyInteger(10, 50)
    num_lstm_layers = fuzzy.FuzzyChoice([1, 2, 3])
    d_model = fuzzy.FuzzyChoice([32, 64, 128])
    num_heads = fuzzy.FuzzyChoice([2, 4, 8])
    output_dim = 1
    hidden_continuous_dim = fuzzy.FuzzyInteger(8, 32)


class TFTLightningFactory(factory.Factory):
    """Factory for creating TemporalFusionTransformer Lightning modules."""

    class Meta:
        model = TemporalFusionTransformer

    num_past_features = fuzzy.FuzzyInteger(2, 5)
    num_future_features = fuzzy.FuzzyInteger(1, 3)
    ctxt_seq_len = fuzzy.FuzzyInteger(50, 200)
    tgt_seq_len = fuzzy.FuzzyInteger(10, 50)
    num_lstm_layers = fuzzy.FuzzyChoice([1, 2, 3])
    d_model = fuzzy.FuzzyChoice([32, 64, 128])
    num_heads = fuzzy.FuzzyChoice([2, 4, 8])
    output_dim = 1
    hidden_continuous_dim = fuzzy.FuzzyInteger(8, 32)
    learning_rate = fuzzy.FuzzyFloat(1e-4, 1e-2)


# Helper functions for creating test data
def create_sample_batch(
    batch_size: int = 4,
    ctxt_seq_len: int = 100,
    tgt_seq_len: int = 50,
    num_features: int = 2,
) -> dict[str, torch.Tensor]:
    """Create a sample batch for testing."""
    encoder_input = torch.randn(batch_size, ctxt_seq_len, num_features)
    decoder_input = torch.randn(batch_size, tgt_seq_len, num_features)

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_lengths": torch.ones(batch_size, 1),
        "decoder_lengths": torch.ones(batch_size, 1),
        "encoder_mask": torch.ones_like(encoder_input),
        "decoder_mask": torch.ones_like(decoder_input),
    }


def create_temp_data_file(
    tmp_path: Path,
    filename: str = "test_data.parquet",
    **kwargs: typing.Any,
) -> str:
    """Create a temporary data file for testing."""
    df = TimeSeriesDataFactory.create(**kwargs)
    filepath = tmp_path / filename
    df.to_parquet(filepath)
    return str(filepath)
