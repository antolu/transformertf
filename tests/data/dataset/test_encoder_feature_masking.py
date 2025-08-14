"""
Test encoder feature masking in EncoderDecoderDataset.

This test ensures that encoder features can be selectively masked (zeroed out)
while decoder features remain untouched.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.data._dataset_factory import DatasetFactory


@pytest.fixture
def test_data():
    """Create synthetic test data with known values."""
    np.random.seed(42)
    return pd.DataFrame({
        "__known_continuous__feature1": np.ones(
            20
        ),  # Will be in both encoder and decoder
        "__known_continuous__feature2": np.ones(20)
        * 2,  # Will be in both encoder and decoder
        "__past_known_continuous__feature3": np.ones(20) * 3,  # Will be in encoder only
        "__target__target": np.ones(20) * 4,  # Target values
    })


def test_encoder_feature_masking_basic(test_data):
    """Test basic encoder feature masking functionality."""
    # Create dataset with masking for one feature
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data,
        ctx_seq_len=10,
        tgt_seq_len=5,
        masked_encoder_features=["__known_continuous__feature1"],
    )

    # Get a sample
    sample = dataset[0]

    # Check that the masked feature is zeroed in encoder
    encoder_input = sample["encoder_input"]

    # Feature1 should be zeroed (assuming it's the first column in encoder)
    # We need to find which column corresponds to feature1
    assert torch.allclose(encoder_input[:, 0], torch.zeros_like(encoder_input[:, 0]))

    # Feature2 and feature3 should not be zeroed
    assert not torch.allclose(
        encoder_input[:, 1], torch.zeros_like(encoder_input[:, 1])
    )
    assert not torch.allclose(
        encoder_input[:, 2], torch.zeros_like(encoder_input[:, 2])
    )

    # Check that decoder input is not affected (should contain feature1 and feature2)
    decoder_input = sample["decoder_input"]
    # Decoder should still have feature1 (not masked)
    assert not torch.allclose(
        decoder_input[:, 0], torch.zeros_like(decoder_input[:, 0])
    )


def test_encoder_feature_masking_multiple(test_data):
    """Test masking multiple encoder features."""
    # Create dataset with masking for multiple features
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data,
        ctx_seq_len=10,
        tgt_seq_len=5,
        masked_encoder_features=[
            "__known_continuous__feature1",
            "__past_known_continuous__feature3",
        ],
    )

    # Get a sample
    sample = dataset[0]
    encoder_input = sample["encoder_input"]

    # Multiple features should be zeroed in encoder
    # (exact column indices depend on data structure)
    num_zero_columns = torch.sum(torch.all(encoder_input == 0, dim=0))
    assert num_zero_columns >= 2  # At least 2 features should be zeroed


def test_encoder_feature_masking_no_masking(test_data):
    """Test that no masking works correctly."""
    # Create dataset with no masking
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data,
        ctx_seq_len=10,
        tgt_seq_len=5,
        masked_encoder_features=None,
    )

    # Get a sample
    sample = dataset[0]
    encoder_input = sample["encoder_input"]

    # No columns should be all zeros (all features preserved)
    num_zero_columns = torch.sum(torch.all(encoder_input == 0, dim=0))
    assert num_zero_columns == 0


def test_encoder_feature_masking_invalid_feature(test_data):
    """Test that invalid feature names are handled gracefully."""
    # Create dataset with invalid feature name
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data,
        ctx_seq_len=10,
        tgt_seq_len=5,
        masked_encoder_features=["nonexistent_feature"],
    )

    # Should not crash, just ignore the invalid feature
    sample = dataset[0]
    assert sample is not None


def test_encoder_feature_masking_empty_list(test_data):
    """Test that empty list for masked features works correctly."""
    # Create dataset with empty masking list
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data,
        ctx_seq_len=10,
        tgt_seq_len=5,
        masked_encoder_features=[],
    )

    # Get a sample
    sample = dataset[0]
    encoder_input = sample["encoder_input"]

    # No columns should be all zeros (all features preserved)
    num_zero_columns = torch.sum(torch.all(encoder_input == 0, dim=0))
    assert num_zero_columns == 0


def test_encoder_feature_masking_preserves_sequence_lengths(test_data):
    """Test that masking doesn't affect sequence length properties."""
    # Create dataset with sequence length randomization and masking
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data,
        ctx_seq_len=10,
        tgt_seq_len=5,
        min_ctx_seq_len=8,
        min_tgt_seq_len=3,
        randomize_seq_len=True,
        masked_encoder_features=["__known_continuous__feature1"],
    )

    # Get a sample
    sample = dataset[0]

    # Check that sequence lengths are still properly set
    assert "encoder_lengths" in sample
    assert "decoder_lengths" in sample
    assert sample["encoder_lengths"].item() >= 8
    assert sample["decoder_lengths"].item() >= 3


def test_encoder_feature_masking_with_different_sequence_lengths():
    """Test masking with different encoder/decoder sequence lengths."""
    # Create data with specific values for easier verification
    test_data_custom = pd.DataFrame({
        "__known_continuous__feature1": [1.0] * 50,
        "__known_continuous__feature2": [2.0] * 50,
        "__past_known_continuous__feature3": [3.0] * 50,
        "__target__target": [4.0] * 50,
    })

    # Test with very different sequence lengths
    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data_custom,
        ctx_seq_len=30,  # Long encoder sequence
        tgt_seq_len=10,  # Short decoder sequence
        masked_encoder_features=["__known_continuous__feature1"],
    )

    sample = dataset[0]

    # Verify shapes
    assert sample["encoder_input"].shape == (30, 4)  # 30 timesteps, 4 features
    assert sample["decoder_input"].shape == (10, 4)  # 10 timesteps, 4 features

    # Verify masking (first feature should be 0 in encoder, 1 in decoder)
    assert torch.allclose(sample["encoder_input"][:, 0], torch.zeros(30))
    assert torch.allclose(sample["decoder_input"][:, 0], torch.ones(10))


def test_encoder_feature_masking_with_transforms():
    """Test that masking works correctly when combined with transforms."""
    # Create test data
    test_data_transforms = pd.DataFrame({
        "__known_continuous__feature1": [5.0] * 30,
        "__known_continuous__feature2": [10.0] * 30,
        "__target__target": [1.0] * 30,
    })

    # Create simple transform that doubles values
    from transformertf.data.transform import BaseTransform

    class DoubleTransform(BaseTransform):
        _transform_type = BaseTransform.TransformType.X

        def fit(self, x, y=None):
            return self

        def transform(self, x, y=None):
            return torch.tensor(x) * 2.0

        def inverse_transform(self, x, y=None):
            return torch.tensor(x) / 2.0

        def __sklearn_is_fitted__(self):  # noqa: PLW3201
            return True

    # Create dataset with transforms and masking
    transforms = {
        "__known_continuous__feature1": DoubleTransform(),
        "__known_continuous__feature2": DoubleTransform(),
    }

    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data_transforms,
        ctx_seq_len=15,
        tgt_seq_len=10,
        transforms=transforms,
        masked_encoder_features=["__known_continuous__feature1"],
    )

    sample = dataset[0]

    # Verify that masking happens AFTER transforms
    # - feature1 should be 0 (masked) in encoder
    # - feature1 should be 10 (5*2, transformed but not masked) in decoder
    # - feature2 should be 20 (10*2, transformed) in both encoder and decoder

    assert torch.allclose(sample["encoder_input"][:, 0], torch.zeros(15))  # Masked
    assert torch.allclose(
        sample["encoder_input"][:, 1], torch.ones(15) * 20
    )  # Transformed
    assert torch.allclose(
        sample["decoder_input"][:, 0], torch.ones(10) * 10
    )  # Transformed, not masked
    assert torch.allclose(
        sample["decoder_input"][:, 1], torch.ones(10) * 20
    )  # Transformed


if __name__ == "__main__":
    # Quick test
    import torch

    test_data_df = pd.DataFrame({
        "__known_continuous__feature1": np.ones(20),
        "__known_continuous__feature2": np.ones(20) * 2,
        "__past_known_continuous__feature3": np.ones(20) * 3,
        "__target__target": np.ones(20) * 4,
    })

    dataset = DatasetFactory.create_encoder_decoder_dataset(
        data=test_data_df,
        ctx_seq_len=10,
        tgt_seq_len=5,
        masked_encoder_features=["__known_continuous__feature1"],
    )

    sample = dataset[0]
    print("Encoder input shape:", sample["encoder_input"].shape)
    print("Decoder input shape:", sample["decoder_input"].shape)
    print("Encoder input (first few rows):")
    print(sample["encoder_input"][:3])
    print("Decoder input (first few rows):")
    print(sample["decoder_input"][:3])
