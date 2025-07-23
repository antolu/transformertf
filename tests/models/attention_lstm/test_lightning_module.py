from __future__ import annotations

import pytest
import torch

from transformertf.models.attention_lstm import AttentionLSTM


def test_attention_lstm_module_training_step() -> None:
    """Test training step of AttentionLSTM."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Create mock batch
    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 3),
        "decoder_input": torch.randn(batch_size, future_seq_len, 2),
        "decoder_lengths": torch.tensor([[future_seq_len]] * batch_size),
        "target": torch.randn(batch_size, future_seq_len, 1),
    }

    # Run training step
    output = model.training_step(batch, batch_idx=0)

    assert isinstance(output, dict)
    assert "loss" in output
    assert "output" in output
    assert "point_prediction" in output
    assert isinstance(output["loss"], torch.Tensor)
    assert output["loss"].dim() == 0  # Scalar loss
    assert torch.isfinite(output["loss"])


def test_attention_lstm_module_validation_step() -> None:
    """Test validation step of AttentionLSTM."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Create mock batch
    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 3),
        "decoder_input": torch.randn(batch_size, future_seq_len, 2),
        "decoder_lengths": torch.tensor([[future_seq_len]] * batch_size),
        "target": torch.randn(batch_size, future_seq_len, 1),
    }

    # Run validation step
    output = model.validation_step(batch, batch_idx=0)

    # validation_step returns a dictionary with output data
    assert isinstance(output, dict)
    assert "loss" in output
    assert "output" in output
    assert "point_prediction" in output


def test_attention_lstm_module_test_step() -> None:
    """Test test step of AttentionLSTM."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Create mock batch
    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 3),
        "decoder_input": torch.randn(batch_size, future_seq_len, 2),
        "decoder_lengths": torch.tensor([[future_seq_len]] * batch_size),
        "target": torch.randn(batch_size, future_seq_len, 1),
    }

    # Run test step
    output = model.test_step(batch, batch_idx=0)

    # test_step returns a dictionary with output data
    assert isinstance(output, dict)
    assert "loss" in output
    assert "output" in output
    assert "point_prediction" in output


def test_attention_lstm_module_predict_step() -> None:
    """Test predict step of AttentionLSTM."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    batch_size = 4
    past_seq_len = 20
    future_seq_len = 10

    # Create mock batch
    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 3),
        "decoder_input": torch.randn(batch_size, future_seq_len, 2),
        "encoder_lengths": torch.tensor([[past_seq_len]] * batch_size),
        "decoder_lengths": torch.tensor([[future_seq_len]] * batch_size),
    }

    # Run predict step
    output = model.predict_step(batch, batch_idx=0)

    assert isinstance(output, dict)
    assert "output" in output
    assert "point_prediction" in output

    point_prediction = output["point_prediction"]
    assert isinstance(point_prediction, torch.Tensor)
    assert point_prediction.shape == (batch_size, future_seq_len, 1)
    assert torch.isfinite(point_prediction).all()


@pytest.mark.xfail(
    reason="configure_optimizers is handled by LightningCLI, not implemented in modules"
)
def test_attention_lstm_module_configure_optimizers() -> None:
    """Test optimizer configuration of AttentionLSTM."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    optimizer_config = model.configure_optimizers()

    assert isinstance(optimizer_config, torch.optim.Optimizer)


def test_attention_lstm_module_hyperparameter_storage() -> None:
    """Test that hyperparameters are properly stored in AttentionLSTM."""
    model = AttentionLSTM(
        num_past_features=5,
        num_future_features=3,
        hidden_size=64,
        num_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    # Check that all hyperparameters are stored
    assert hasattr(model, "hparams")
    assert model.hparams.num_past_features == 5
    assert model.hparams.num_future_features == 3
    assert model.hparams.hidden_size == 64
    assert model.hparams.num_layers == 2
    assert model.hparams.n_heads == 4
    assert model.hparams.dropout == 0.1
    assert model.hparams.use_gating is True


def test_attention_lstm_module_model_instantiation() -> None:
    """Test that the underlying model is properly instantiated."""
    lightning_module = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=True,
    )

    # Check that the model attribute exists and is of correct type
    assert hasattr(lightning_module, "model")
    assert lightning_module.model is not None

    # Check that model has the correct parameters
    assert lightning_module.model.num_past_features == 3
    assert lightning_module.model.num_future_features == 2
    assert lightning_module.model.hidden_size == 32
    assert lightning_module.model.num_layers == 1
    assert lightning_module.model.n_heads == 2
    assert lightning_module.model.dropout == 0.1
    assert lightning_module.model.use_gating is True


@pytest.mark.parametrize("use_gating", [True, False])
def test_attention_lstm_module_gating_options(use_gating: bool) -> None:
    """Test AttentionLSTM with different gating options."""
    model = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        use_gating=use_gating,
    )

    assert model.hparams.use_gating == use_gating
    assert model.model.use_gating == use_gating

    # Test forward pass works with both options
    batch_size = 2
    past_seq_len = 10
    future_seq_len = 5

    batch = {
        "encoder_input": torch.randn(batch_size, past_seq_len, 3),
        "decoder_input": torch.randn(batch_size, future_seq_len, 2),
        "decoder_lengths": torch.tensor([[future_seq_len]] * batch_size),
        "target": torch.randn(batch_size, future_seq_len, 1),
    }

    step_output = model.training_step(batch, batch_idx=0)
    assert isinstance(step_output, dict)
    assert "loss" in step_output
    assert torch.isfinite(step_output["loss"])
