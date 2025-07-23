from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.attention_lstm import AttentionLSTM


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample time series data for testing."""
    n_samples = 1000
    n_features = 5

    # Generate synthetic time series data
    data = []
    for i in range(n_samples):
        row = {
            "time": i,
            "group": "test_group",
            "target": torch.randn(1).item(),
        }
        # Add feature columns
        for j in range(n_features):
            row[f"feature_{j}"] = torch.randn(1).item()
        data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_data: pd.DataFrame) -> Path:
    """Create temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = Path(f.name)
        sample_data.to_parquet(temp_path)
        return temp_path


@pytest.mark.xfail(
    reason="Complex data module integration requires proper data format and group configuration"
)
def test_attention_lstm_with_data_module(temp_data_file: Path) -> None:
    """Test AttentionLSTM integration with EncoderDecoderDataModule."""
    # Create data module
    datamodule = EncoderDecoderDataModule(
        train_df_paths=[str(temp_data_file)],
        val_df_paths=[str(temp_data_file)],
        time_column="time",
        target_covariate="target",
        known_covariates=["feature_0", "feature_1", "feature_2"],
        ctxt_seq_len=50,
        tgt_seq_len=10,
        batch_size=4,
        num_workers=0,
    )

    # Setup the data module
    datamodule.setup()

    # Get feature dimensions from data module
    num_past_features = datamodule.num_past_known_covariates
    num_future_features = datamodule.num_future_known_covariates

    # Create model
    model = AttentionLSTM(
        num_past_features=num_past_features,
        num_future_features=num_future_features,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    # Test with train dataloader
    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))

    # Test training step
    output = model.training_step(batch, batch_idx=0)
    assert isinstance(output, dict)
    assert "loss" in output
    assert torch.isfinite(output["loss"])

    # Test validation step
    val_dataloader = datamodule.val_dataloader()
    val_batch = next(iter(val_dataloader))
    model.validation_step(val_batch, batch_idx=0)

    # Test predict step
    predictions = model.predict_step(val_batch, batch_idx=0)
    assert isinstance(predictions, dict)
    assert "point_prediction" in predictions
    assert (
        predictions["point_prediction"].shape[0] == val_batch["target"].shape[0]
    )  # Same batch size

    # Clean up
    temp_data_file.unlink()


def test_attention_lstm_parameter_compatibility() -> None:
    """Test that AttentionLSTM parameters are compatible with CLI linking."""
    # This test verifies that the model can be instantiated with parameters
    # that would be linked from the data module in the CLI

    # Simulate parameters that would come from data module
    data_params = {
        "num_past_features": 5,
        "num_future_features": 3,
        "ctxt_seq_len": 40,
        "tgt_seq_len": 10,
    }

    # Create model with these parameters
    model = AttentionLSTM(
        num_past_features=data_params["num_past_features"],
        num_future_features=data_params["num_future_features"],
        hidden_size=64,
        num_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    # Verify parameters are set correctly
    assert model.hparams.num_past_features == data_params["num_past_features"]
    assert model.hparams.num_future_features == data_params["num_future_features"]

    # Test forward pass with appropriate tensor sizes
    batch_size = 4
    past_covariates = torch.randn(
        batch_size, data_params["ctxt_seq_len"], data_params["num_past_features"]
    )
    future_covariates = torch.randn(
        batch_size, data_params["tgt_seq_len"], data_params["num_future_features"]
    )
    decoder_lengths = torch.tensor(
        [[data_params["tgt_seq_len"]]] * batch_size
    )  # (B, 1) shape

    batch = {
        "encoder_input": past_covariates,
        "decoder_input": future_covariates,
        "decoder_lengths": decoder_lengths,
        "target": torch.randn(batch_size, data_params["tgt_seq_len"], 1),
    }

    step_output = model.training_step(batch, batch_idx=0)
    assert isinstance(step_output, dict)
    assert "loss" in step_output
    assert torch.isfinite(step_output["loss"])


def test_attention_lstm_minimal_integration() -> None:
    """Test minimal integration scenario with smallest possible configuration."""
    # Create minimal model
    model = AttentionLSTM(
        num_past_features=2,
        num_future_features=1,
        hidden_size=16,
        num_layers=1,
        n_heads=1,
        dropout=0.0,
        use_gating=False,
    )

    # Create minimal batch
    batch_size = 2
    past_seq_len = 5
    future_seq_len = 3

    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 2),
        "decoder_input": torch.randn(batch_size, future_seq_len, 1),
        "decoder_lengths": torch.tensor(
            [[future_seq_len]] * batch_size
        ),  # (B, 1) shape
        "target": torch.randn(batch_size, future_seq_len, 1),
    }

    # Test all step types
    train_output = model.training_step(batch, batch_idx=0)
    assert isinstance(train_output, dict)
    assert "loss" in train_output
    assert torch.isfinite(train_output["loss"])

    model.validation_step(batch, batch_idx=0)
    model.test_step(batch, batch_idx=0)

    predictions = model.predict_step(batch, batch_idx=0)
    assert isinstance(predictions, dict)
    assert "point_prediction" in predictions
    assert predictions["point_prediction"].shape == (batch_size, future_seq_len, 1)
    assert torch.isfinite(predictions["point_prediction"]).all()


@pytest.mark.parametrize("compile_model", [True, False])
def test_attention_lstm_compile_option(compile_model: bool) -> None:
    """Test AttentionLSTM with compile option."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
        compile_model=compile_model,
    )

    assert model.hparams.compile_model == compile_model

    # Test that model works regardless of compile option
    batch_size = 2
    past_seq_len = 10
    future_seq_len = 5

    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 3),
        "decoder_input": torch.randn(batch_size, future_seq_len, 2),
        "decoder_lengths": torch.tensor([[future_seq_len]] * batch_size),
        "target": torch.randn(batch_size, future_seq_len, 1),
    }

    output = model.training_step(batch, batch_idx=0)
    assert isinstance(output, dict)
    assert "loss" in output
    assert torch.isfinite(output["loss"])
