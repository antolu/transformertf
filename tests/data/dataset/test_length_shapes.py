from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.data.dataset import (
    EncoderDecoderDataset,
    EncoderDecoderPredictDataset,
)
from transformertf.data.transform import TransformCollection


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "input": np.arange(10, dtype=float),
        "target": np.arange(10, 20, dtype=float),
        "optional": np.arange(20, 30, dtype=float),
    })


@pytest.fixture
def basic_transforms():
    """Basic transforms for testing."""
    return {
        "input": TransformCollection([]),
        "target": TransformCollection([]),
    }


def test_encoder_decoder_dataset_length_shapes(sample_data, basic_transforms):
    """Test that EncoderDecoderDataset produces consistent length shapes."""
    dataset = EncoderDecoderDataset(
        input_data=sample_data[["input"]],
        target_data=sample_data[["target"]],
        ctx_seq_len=4,
        tgt_seq_len=3,
        transforms=basic_transforms,
    )

    sample = dataset[0]

    # Both lengths should be Series
    assert isinstance(sample["encoder_lengths"], torch.Tensor)
    assert isinstance(sample["decoder_lengths"], torch.Tensor)

    # Should be 1D tensors with single values
    assert sample["encoder_lengths"].dim() == 1
    assert sample["decoder_lengths"].dim() == 1
    assert len(sample["encoder_lengths"]) == 1
    assert len(sample["decoder_lengths"]) == 1

    # Values should match expected lengths
    assert int(sample["encoder_lengths"].item()) == 4
    assert int(sample["decoder_lengths"].item()) <= 3  # Could be randomized


def test_predict_dataset_length_shapes(sample_data, basic_transforms):
    """Test that EncoderDecoderPredictDataset produces consistent length shapes."""
    past_covariates = sample_data[["input"]].iloc[:4]
    future_covariates = sample_data[["input"]].iloc[4:7]
    past_target = sample_data[["target"]].iloc[:4]

    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=4,
        prediction_length=3,
        input_columns=["input"],
        target_column="target",
        transforms=basic_transforms,
    )

    sample = dataset[0]

    # Both lengths should be tensors (converted from Series)
    assert isinstance(sample["encoder_lengths"], torch.Tensor)
    assert "decoder_lengths" not in sample or isinstance(
        sample["decoder_lengths"], torch.Tensor
    )

    # Should be 1D tensors with single values
    assert sample["encoder_lengths"].dim() == 1
    assert len(sample["encoder_lengths"]) == 1

    # Values should match expected lengths
    assert int(sample["encoder_lengths"].item()) == 4


def test_length_extraction_helper_dataframe():
    """Test _extract_length_value helper with DataFrame input."""
    from transformertf.data.dataset._encoder_decoder import _extract_length_value

    # Valid DataFrame (1x1)
    df = pd.DataFrame({"length": [5]})
    assert _extract_length_value(df) == 5

    # Invalid DataFrame shape
    df_invalid = pd.DataFrame({"length": [5, 6]})
    with pytest.raises(ValueError, match="Expected DataFrame shape \\(1, 1\\)"):
        _extract_length_value(df_invalid)


def test_length_extraction_helper_series():
    """Test _extract_length_value helper with Series input."""
    from transformertf.data.dataset._encoder_decoder import _extract_length_value

    # Valid Series (length 1)
    series = pd.Series([5], name="length")
    assert _extract_length_value(series) == 5

    # Invalid Series length
    series_invalid = pd.Series([5, 6], name="length")
    with pytest.raises(ValueError, match="Expected Series length 1"):
        _extract_length_value(series_invalid)


def test_static_method_length_shapes(sample_data):
    """Test that static methods produce Series for lengths."""
    encoder_sample = EncoderDecoderDataset.make_encoder_input(
        sample_data[["input"]].iloc[:4]
    )
    decoder_sample = EncoderDecoderDataset.make_decoder_input(
        sample_data[["target"]].iloc[:3]
    )

    # Both should produce Series for lengths (before tensor conversion)
    # Note: static methods now convert to tensors directly
    assert isinstance(encoder_sample["encoder_lengths"], torch.Tensor)
    assert isinstance(decoder_sample["decoder_lengths"], torch.Tensor)

    # Tensors should be 1D with single values
    assert encoder_sample["encoder_lengths"].dim() == 1
    assert decoder_sample["decoder_lengths"].dim() == 1
    assert len(encoder_sample["encoder_lengths"]) == 1
    assert len(decoder_sample["decoder_lengths"]) == 1

    # Values should match input lengths
    assert int(encoder_sample["encoder_lengths"].item()) == 4
    assert int(decoder_sample["decoder_lengths"].item()) == 3


def test_cross_dataset_length_consistency(sample_data, basic_transforms):
    """Test that both dataset types handle lengths consistently."""
    # Regular dataset
    regular_dataset = EncoderDecoderDataset(
        input_data=sample_data[["input"]],
        target_data=sample_data[["target"]],
        ctx_seq_len=4,
        tgt_seq_len=3,
        transforms=basic_transforms,
        randomize_seq_len=False,  # Disable randomization for predictable lengths
    )

    # Predict dataset
    past_covariates = sample_data[["input"]].iloc[:4]
    future_covariates = sample_data[["input"]].iloc[4:7]
    past_target = sample_data[["target"]].iloc[:4]

    predict_dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=4,
        prediction_length=3,
        input_columns=["input"],
        target_column="target",
        transforms=basic_transforms,
    )

    regular_sample = regular_dataset[0]
    predict_sample = predict_dataset[0]

    # Both should have 1D tensors for lengths
    assert regular_sample["encoder_lengths"].dim() == 1
    assert predict_sample["encoder_lengths"].dim() == 1

    # Both should have same encoder length
    assert (
        regular_sample["encoder_lengths"].item()
        == predict_sample["encoder_lengths"].item()
    )


def test_no_iloc_double_indexing_errors(sample_data, basic_transforms):
    """Ensure no double indexing errors like .iloc[0, 0] on Series."""
    dataset = EncoderDecoderDataset(
        input_data=sample_data[["input"]],
        target_data=sample_data[["target"]],
        ctx_seq_len=4,
        tgt_seq_len=3,
        transforms=basic_transforms,
    )

    # This should not raise "Too many indexers" error
    sample = dataset[0]

    # Verify we can access the length values without errors
    encoder_len = int(sample["encoder_lengths"].item())
    decoder_len = int(sample["decoder_lengths"].item())

    assert encoder_len == 4
    assert decoder_len <= 3


def test_length_tensor_devices_and_dtypes(sample_data, basic_transforms):
    """Test that length tensors have correct device and dtype."""
    dataset = EncoderDecoderDataset(
        input_data=sample_data[["input"]],
        target_data=sample_data[["target"]],
        ctx_seq_len=4,
        tgt_seq_len=3,
        transforms=basic_transforms,
        dtype=torch.float32,
    )

    sample = dataset[0]

    # Check that length tensors match main tensor device and dtype
    assert sample["encoder_lengths"].device == sample["encoder_input"].device
    assert sample["decoder_lengths"].device == sample["decoder_input"].device
    assert sample["encoder_lengths"].dtype == sample["encoder_input"].dtype
    assert sample["decoder_lengths"].dtype == sample["decoder_input"].dtype
