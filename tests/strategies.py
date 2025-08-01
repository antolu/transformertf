"""Hypothesis strategies for property-based testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

# Basic strategies for common types
positive_integers = st.integers(min_value=1, max_value=1000)
small_positive_integers = st.integers(min_value=1, max_value=20)
sequence_lengths = st.integers(min_value=10, max_value=500)
batch_sizes = st.integers(min_value=1, max_value=64)
feature_dimensions = st.integers(min_value=1, max_value=10)
learning_rates = st.floats(min_value=1e-5, max_value=1e-1)
dropout_rates = st.floats(min_value=0.0, max_value=0.9)


# Tensor strategies
@st.composite
def tensor_strategy(
    draw,
    shape=None,
    dtype=torch.float32,
    min_value=-10.0,
    max_value=10.0,
    finite_only=True,
):
    """Strategy for generating PyTorch tensors."""
    if shape is None:
        # Generate random shape
        ndims = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(st.integers(min_value=1, max_value=100)) for _ in range(ndims)
        )

    # Generate numpy array first
    np_array = draw(
        np_arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=not finite_only,
                allow_infinity=not finite_only,
                width=32,  # Use 32-bit precision to match dtype
            ),
        )
    )

    return torch.from_numpy(np_array).to(dtype)


# Time series strategies
@st.composite
def time_series_strategy(draw, min_length=50, max_length=1000, n_features=None):
    """Strategy for generating time series data."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    if n_features is None:
        n_features = draw(st.integers(min_value=1, max_value=5))

    # Generate time column
    time_data = np.linspace(0, 100, length)

    # Generate feature columns
    columns_data = {"time_ms": time_data * 1000}

    for i in range(n_features):
        # Generate realistic time series with trend and noise
        trend = draw(st.floats(min_value=-0.1, max_value=0.1))
        noise_std = draw(st.floats(min_value=0.01, max_value=0.5))

        values = trend * time_data + np.random.normal(0, noise_std, length)
        columns_data[f"feature_{i}"] = values

    return pd.DataFrame(columns_data)


# Model configuration strategies
@st.composite
def tft_config_strategy(draw):
    """Strategy for generating TFT model configurations."""
    return {
        "num_past_features": draw(st.integers(min_value=1, max_value=5)),
        "num_future_features": draw(st.integers(min_value=1, max_value=3)),
        "ctxt_seq_len": draw(st.integers(min_value=10, max_value=200)),
        "tgt_seq_len": draw(st.integers(min_value=5, max_value=100)),
        "num_lstm_layers": draw(st.integers(min_value=1, max_value=3)),
        "d_model": draw(st.sampled_from([16, 32, 64, 128])),
        "num_heads": draw(st.sampled_from([1, 2, 4, 8])),
        "output_dim": 1,
        "hidden_continuous_dim": draw(st.integers(min_value=8, max_value=32)),
        "dropout": draw(st.floats(min_value=0.0, max_value=0.5)),
    }


@st.composite
def datamodule_config_strategy(draw):
    """Strategy for generating DataModule configurations."""
    return {
        "ctxt_seq_len": draw(st.integers(min_value=10, max_value=200)),
        "tgt_seq_len": draw(st.integers(min_value=5, max_value=100)),
        "batch_size": draw(st.integers(min_value=1, max_value=32)),
        "stride": draw(st.integers(min_value=1, max_value=10)),
        "downsample": draw(st.integers(min_value=1, max_value=5)),
        "normalize": draw(st.booleans()),
        "num_workers": 0,  # Keep at 0 for testing
    }


# Transform strategies
@st.composite
def polynomial_coefficients_strategy(draw, degree=None):
    """Strategy for generating polynomial coefficients."""
    if degree is None:
        degree = draw(st.integers(min_value=1, max_value=5))

    coefficients = []
    for _ in range(degree + 1):
        coeff = draw(st.floats(min_value=-10.0, max_value=10.0))
        coefficients.append(coeff)

    return coefficients


@st.composite
def scaling_parameters_strategy(draw):
    """Strategy for generating scaling parameters."""
    return {
        "scale": draw(st.floats(min_value=0.1, max_value=10.0)),
        "offset": draw(st.floats(min_value=-10.0, max_value=10.0)),
    }


# Batch strategies
@st.composite
def encoder_decoder_batch_strategy(draw):
    """Strategy for generating encoder-decoder batches."""
    batch_size = draw(st.integers(min_value=1, max_value=16))
    ctxt_seq_len = draw(st.integers(min_value=10, max_value=100))
    tgt_seq_len = draw(st.integers(min_value=5, max_value=50))
    num_features = draw(st.integers(min_value=1, max_value=5))

    encoder_input = draw(
        tensor_strategy(shape=(batch_size, ctxt_seq_len, num_features))
    )
    decoder_input = draw(tensor_strategy(shape=(batch_size, tgt_seq_len, num_features)))

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_lengths": torch.ones(batch_size, 1),
        "decoder_lengths": torch.ones(batch_size, 1),
        "encoder_mask": torch.ones_like(encoder_input),
        "decoder_mask": torch.ones_like(decoder_input),
    }


# Attention strategies
@st.composite
def attention_config_strategy(draw):
    """Strategy for generating attention configurations."""
    n_dim_model = draw(st.sampled_from([16, 32, 64, 128]))
    n_heads = draw(st.sampled_from([1, 2, 4, 8]))

    # Ensure n_dim_model is divisible by n_heads
    while n_dim_model % n_heads != 0:
        n_heads = draw(st.sampled_from([1, 2, 4, 8]))

    return {
        "n_dim_model": n_dim_model,
        "n_heads": n_heads,
        "dropout": draw(st.floats(min_value=0.0, max_value=0.5)),
    }


# Loss function strategies
@st.composite
def quantile_strategy(draw):
    """Strategy for generating quantile values."""
    # Generate sorted quantiles between 0 and 1
    n_quantiles = draw(st.integers(min_value=1, max_value=9))
    quantiles = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=0.99),
            min_size=n_quantiles,
            max_size=n_quantiles,
        )
    )
    return sorted(quantiles)


# Optimization strategies
@st.composite
def optimizer_config_strategy(draw):
    """Strategy for generating optimizer configurations."""
    optimizer_type = draw(st.sampled_from(["adam", "sgd", "adamw"]))

    config = {
        "lr": draw(st.floats(min_value=1e-5, max_value=1e-1)),
        "weight_decay": draw(st.floats(min_value=0.0, max_value=1e-3)),
    }

    if optimizer_type == "sgd":
        config["momentum"] = draw(st.floats(min_value=0.0, max_value=0.99))
    elif optimizer_type in ["adam", "adamw"]:
        config["betas"] = (
            draw(st.floats(min_value=0.5, max_value=0.99)),
            draw(st.floats(min_value=0.9, max_value=0.999)),
        )

    return optimizer_type, config


# TCT (Temporal Convolutional Transformer) strategies
@st.composite
def tct_config_strategy(draw):
    """Strategy for generating TCT model configurations."""
    # Ensure sufficient sequence lengths for compression
    compression_factor = draw(st.sampled_from([2, 4, 8]))
    max_dilation = draw(st.sampled_from([4, 8, 16]))

    # Calculate minimum sequence lengths based on compression requirements
    min_encoder_len = compression_factor * max_dilation * 12
    min_decoder_len = compression_factor * 8

    return {
        "num_past_features": draw(st.integers(min_value=2, max_value=8)),
        "num_future_features": draw(st.integers(min_value=1, max_value=6)),
        "output_dim": draw(st.sampled_from([1, 3, 5])),
        "hidden_dim": draw(st.sampled_from([32, 64, 128])),
        "num_attention_heads": draw(st.sampled_from([2, 4, 8])),
        "compression_factor": compression_factor,
        "num_encoder_layers": draw(st.integers(min_value=2, max_value=6)),
        "num_decoder_layers": draw(st.integers(min_value=2, max_value=6)),
        "dropout": draw(st.floats(min_value=0.0, max_value=0.3)),
        "max_dilation": max_dilation,
        "min_encoder_len": min_encoder_len,
        "min_decoder_len": min_decoder_len,
    }


@st.composite
def tct_batch_strategy(draw):
    """Strategy for generating TCT-compatible encoder-decoder batches."""
    batch_size = draw(st.integers(min_value=1, max_value=8))

    # Get TCT config to ensure sufficient sequence lengths
    tct_config = draw(tct_config_strategy())

    # Use minimum lengths plus some buffer
    encoder_len = draw(
        st.integers(
            min_value=tct_config["min_encoder_len"],
            max_value=tct_config["min_encoder_len"] + 200,
        )
    )
    decoder_len = draw(
        st.integers(
            min_value=tct_config["min_decoder_len"],
            max_value=tct_config["min_decoder_len"] + 100,
        )
    )

    num_past_features = tct_config["num_past_features"]
    num_future_features = tct_config["num_future_features"]

    encoder_input = draw(
        tensor_strategy(shape=(batch_size, encoder_len, num_past_features))
    )
    decoder_input = draw(
        tensor_strategy(shape=(batch_size, decoder_len, num_future_features))
    )
    target = draw(tensor_strategy(shape=(batch_size, decoder_len, 1)))

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "target": target,
        "encoder_lengths": torch.full((batch_size, 1), encoder_len),
        "decoder_lengths": torch.full((batch_size, 1), decoder_len),
        "config": tct_config,
    }


@st.composite
def long_sequence_strategy(draw, min_compression_factor=2):
    """Strategy for generating sequences that meet TCT compression requirements."""
    compression_factor = draw(
        st.integers(min_value=min_compression_factor, max_value=8)
    )
    max_dilation = draw(st.integers(min_value=4, max_value=16))

    # Calculate minimum lengths
    min_encoder_len = compression_factor * max_dilation * 12
    min_decoder_len = compression_factor * 8

    # Generate lengths with buffer above minimum
    encoder_len = draw(
        st.integers(min_value=min_encoder_len, max_value=min_encoder_len + 400)
    )
    decoder_len = draw(
        st.integers(min_value=min_decoder_len, max_value=min_decoder_len + 200)
    )

    return {
        "encoder_len": encoder_len,
        "decoder_len": decoder_len,
        "compression_factor": compression_factor,
        "max_dilation": max_dilation,
    }


# Composite strategies for full model testing
@st.composite
def full_model_setup_strategy(draw):
    """Strategy for generating complete model setup."""
    model_config = draw(tft_config_strategy())
    datamodule_config = draw(datamodule_config_strategy())

    # Ensure consistency between model and datamodule
    datamodule_config["ctxt_seq_len"] = model_config["ctxt_seq_len"]
    datamodule_config["tgt_seq_len"] = model_config["tgt_seq_len"]

    return {
        "model_config": model_config,
        "datamodule_config": datamodule_config,
        "optimizer_config": draw(optimizer_config_strategy()),
    }


@st.composite
def full_tct_setup_strategy(draw):
    """Strategy for generating complete TCT model setup."""
    tct_config = draw(tct_config_strategy())

    # Create datamodule config compatible with TCT requirements
    datamodule_config = {
        "ctxt_seq_len": tct_config["min_encoder_len"],
        "tgt_seq_len": tct_config["min_decoder_len"],
        "batch_size": draw(st.integers(min_value=2, max_value=16)),
        "stride": draw(st.integers(min_value=1, max_value=5)),
        "downsample": 1,  # No downsampling for TCT tests
        "normalize": draw(st.booleans()),
        "num_workers": 0,
    }

    return {
        "tct_config": tct_config,
        "datamodule_config": datamodule_config,
        "optimizer_config": draw(optimizer_config_strategy()),
    }


# Edge case strategies
@st.composite
def edge_case_tensor_strategy(draw):
    """Strategy for generating edge case tensors."""
    case_type = draw(
        st.sampled_from(["zeros", "ones", "very_small", "very_large", "mixed_signs"])
    )

    shape = draw(
        st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10),
        )
    )

    if case_type == "zeros":
        return torch.zeros(shape)
    if case_type == "ones":
        return torch.ones(shape)
    if case_type == "very_small":
        return torch.full(shape, 1e-8)
    if case_type == "very_large":
        return torch.full(shape, 1e8)
    if case_type == "mixed_signs":
        tensor = torch.randn(shape)
        # Ensure we have both positive and negative values
        tensor[0, 0] = 1.0
        tensor[0, 1] = -1.0
        return tensor

    return torch.randn(shape)


# Validation strategies
def valid_sequence_length():
    """Strategy for valid sequence lengths."""
    return st.integers(min_value=1, max_value=1000)


def valid_batch_size():
    """Strategy for valid batch sizes."""
    return st.integers(min_value=1, max_value=128)


def valid_feature_dimension():
    """Strategy for valid feature dimensions."""
    return st.integers(min_value=1, max_value=20)
