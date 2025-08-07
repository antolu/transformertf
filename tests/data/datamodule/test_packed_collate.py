from __future__ import annotations

import torch

from transformertf.data.datamodule._transformer import EncoderDecoderDataModule
from transformertf.models.attention_lstm import AttentionLSTMModel


def test_collate_with_right_alignment():
    """Test collate function behavior with right alignment (TFT-style)."""
    # Create datamodule with right alignment
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1", "feature2"],
        target_covariate="target",
        encoder_alignment="right",
    )

    # Create samples
    samples = [
        {
            "encoder_input": torch.randn(10, 4),
            "decoder_input": torch.randn(8, 3),
            "target": torch.randn(8, 1),
            "encoder_lengths": torch.tensor(8),
            "decoder_lengths": torch.tensor(6),
        },
        {
            "encoder_input": torch.randn(12, 4),
            "decoder_input": torch.randn(7, 3),
            "target": torch.randn(7, 1),
            "encoder_lengths": torch.tensor(10),
            "decoder_lengths": torch.tensor(5),
        },
    ]

    # Collate batch with right alignment
    batch = datamodule.collate_fn()(samples)

    # Should use standard TFT behavior (no left alignment)
    assert batch["encoder_input"].shape == (2, 10, 4)  # Trimmed to max length
    assert batch["decoder_input"].shape == (2, 6, 3)
    assert batch["target"].shape == (2, 6, 1)


def test_collate_with_left_alignment():
    """Test collate function behavior with left alignment (LSTM-style)."""
    # Create datamodule with left alignment
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1", "feature2"],
        target_covariate="target",
        encoder_alignment="left",
    )

    # Create samples with same tensor sizes (as real DataModule would)
    max_enc_len, max_dec_len = 12, 8
    samples = [
        {
            "encoder_input": torch.randn(max_enc_len, 4),
            "decoder_input": torch.randn(max_dec_len, 3),
            "target": torch.randn(max_dec_len, 1),
            "encoder_lengths": torch.tensor(8),
            "decoder_lengths": torch.tensor(6),
        },
        {
            "encoder_input": torch.randn(max_enc_len, 4),
            "decoder_input": torch.randn(max_dec_len, 3),
            "target": torch.randn(max_dec_len, 1),
            "encoder_lengths": torch.tensor(10),
            "decoder_lengths": torch.tensor(5),
        },
        {
            "encoder_input": torch.randn(max_enc_len, 4),
            "decoder_input": torch.randn(max_dec_len, 3),
            "target": torch.randn(max_dec_len, 1),
            "encoder_lengths": torch.tensor(7),
            "decoder_lengths": torch.tensor(4),
        },
    ]

    # Collate batch with left alignment
    batch = datamodule.collate_fn()(samples)

    # Should align encoder sequences for packing when lengths differ
    # The collate function trims to the max length needed
    assert batch["encoder_input"].shape == (
        3,
        10,
        4,
    )  # Trimmed to max encoder length needed (10)
    assert batch["decoder_input"].shape == (
        3,
        6,
        3,
    )  # Trimmed to max decoder length needed (6)
    assert batch["target"].shape == (3, 6, 1)  # Same as decoder

    # Verify encoder lengths are preserved
    expected_encoder_lengths = torch.tensor([8, 10, 7])
    assert torch.equal(batch["encoder_lengths"].squeeze(), expected_encoder_lengths)

    # Verify decoder lengths are preserved
    expected_decoder_lengths = torch.tensor([6, 5, 4])
    assert torch.equal(batch["decoder_lengths"].squeeze(), expected_decoder_lengths)


def test_collate_with_uniform_lengths():
    """Test that uniform lengths don't trigger alignment even with left alignment."""
    # Create datamodule with left alignment
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1", "feature2"],
        target_covariate="target",
        encoder_alignment="left",
    )

    # Create samples with uniform lengths
    samples = [
        {
            "encoder_input": torch.randn(10, 4),
            "decoder_input": torch.randn(8, 3),
            "target": torch.randn(8, 1),
            "encoder_lengths": torch.tensor(10),  # All same length
            "decoder_lengths": torch.tensor(8),
        },
        {
            "encoder_input": torch.randn(10, 4),
            "decoder_input": torch.randn(8, 3),
            "target": torch.randn(8, 1),
            "encoder_lengths": torch.tensor(10),
            "decoder_lengths": torch.tensor(8),
        },
    ]

    # Collate batch
    batch = datamodule.collate_fn()(samples)

    # Standard collation should occur (no alignment needed for uniform lengths)
    assert batch["encoder_input"].shape == (2, 10, 4)
    assert batch["decoder_input"].shape == (2, 8, 3)
    assert batch["target"].shape == (2, 8, 1)


def test_collate_with_masks():
    """Test collate function preserves masks when aligning sequences."""
    # Create datamodule with left alignment
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1"],
        target_covariate="target",
        encoder_alignment="left",
    )

    # Create samples with masks and variable lengths
    samples = [
        {
            "encoder_input": torch.randn(8, 2),
            "decoder_input": torch.randn(6, 2),
            "target": torch.randn(6, 1),
            "encoder_lengths": torch.tensor(6),
            "decoder_lengths": torch.tensor(4),
            "encoder_mask": torch.ones(8, 2),
            "decoder_mask": torch.ones(6, 2),
        },
        {
            "encoder_input": torch.randn(10, 2),
            "decoder_input": torch.randn(5, 2),
            "target": torch.randn(5, 1),
            "encoder_lengths": torch.tensor(8),
            "decoder_lengths": torch.tensor(3),
            "encoder_mask": torch.ones(10, 2),
            "decoder_mask": torch.ones(5, 2),
        },
    ]

    # Collate batch
    batch = datamodule.collate_fn()(samples)

    # Verify shapes
    assert batch["encoder_input"].shape == (2, 8, 2)
    assert batch["encoder_mask"].shape == (2, 8, 2)
    assert batch["decoder_input"].shape == (2, 4, 2)
    assert batch["decoder_mask"].shape == (2, 4, 2)


def test_end_to_end_integration():
    """Test complete integration from collate to model forward pass."""
    # Create datamodule and model
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1", "feature2"],
        target_covariate="target",
        encoder_alignment="left",
    )
    model = AttentionLSTMModel(
        num_past_features=3,
        num_future_features=2,
        d_model=16,
        num_layers=1,
        num_heads=2,
    )

    # Create samples with consistent tensor sizes
    max_enc_len, max_dec_len = 14, 8
    samples = [
        {
            "encoder_input": torch.randn(max_enc_len, 3),
            "decoder_input": torch.randn(max_dec_len, 2),
            "target": torch.randn(max_dec_len, 1),
            "encoder_lengths": torch.tensor(10),
            "decoder_lengths": torch.tensor(6),
        },
        {
            "encoder_input": torch.randn(max_enc_len, 3),
            "decoder_input": torch.randn(max_dec_len, 2),
            "target": torch.randn(max_dec_len, 1),
            "encoder_lengths": torch.tensor(8),
            "decoder_lengths": torch.tensor(5),
        },
        {
            "encoder_input": torch.randn(max_enc_len, 3),
            "decoder_input": torch.randn(max_dec_len, 2),
            "target": torch.randn(max_dec_len, 1),
            "encoder_lengths": torch.tensor(12),
            "decoder_lengths": torch.tensor(4),
        },
    ]

    # Collate batch (should apply alignment for LSTM models)
    batch = datamodule.collate_fn()(samples)

    # Forward pass through model
    with torch.no_grad():
        output = model(
            batch["encoder_input"],
            batch["decoder_input"],
            batch["encoder_lengths"].squeeze(),
            batch["decoder_lengths"].squeeze(),
        )

    # Verify output shape and validity
    expected_shape = (3, batch["decoder_input"].size(1), model.output_dim)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_picklability():
    """Test that the collate function remains picklable."""
    import pickle

    # Create datamodule
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1"],
        target_covariate="target",
        encoder_alignment="left",
    )

    # The collate function should be picklable
    collate_fn = datamodule.collate_fn

    try:
        # This should not raise an exception
        pickled = pickle.dumps(collate_fn)
        unpickled = pickle.loads(pickled)

        # Should be callable
        assert callable(unpickled)
    except Exception as e:
        msg = f"Collate function is not picklable: {e}"
        raise AssertionError(msg) from e


def test_large_batch_handling():
    """Test that the collate function handles larger batches correctly."""
    # Create datamodule with left alignment
    datamodule = EncoderDecoderDataModule(
        known_covariates=["feature1", "feature2"],
        target_covariate="target",
        encoder_alignment="left",
    )

    # Create a larger batch with variable lengths
    batch_size = 32
    samples = [
        {
            "encoder_input": torch.randn(15 + i % 5, 4),
            "decoder_input": torch.randn(10 - i % 3, 3),
            "target": torch.randn(10 - i % 3, 1),
            "encoder_lengths": torch.tensor(12 + i % 4),
            "decoder_lengths": torch.tensor(8 - i % 2),
        }
        for i in range(batch_size)
    ]

    # Collate batch
    batch = datamodule.collate_fn()(samples)

    # Verify correct batch shape
    assert batch["encoder_input"].shape[0] == batch_size
    assert batch["decoder_input"].shape[0] == batch_size
    assert batch["target"].shape[0] == batch_size
