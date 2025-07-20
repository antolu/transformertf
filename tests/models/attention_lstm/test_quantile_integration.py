from __future__ import annotations

import torch

from transformertf.models.attention_lstm import AttentionLSTM
from transformertf.nn import QuantileLoss


def test_attention_lstm_module_quantile_loss_basic() -> None:
    """Test AttentionLSTM with QuantileLoss for probabilistic forecasting."""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    criterion = QuantileLoss(quantiles=quantiles)

    module = AttentionLSTM(
        num_past_features=5,
        num_future_features=3,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.1,
        criterion=criterion,
    )

    # Check that output_dim was automatically set to number of quantiles
    assert module.hparams.output_dim == len(quantiles)
    assert module.model.output_dim == len(quantiles)

    # Test forward pass
    batch = {
        "encoder_input": torch.randn(4, 20, 5),
        "decoder_input": torch.randn(4, 10, 3),
        "decoder_lengths": torch.tensor([10, 10, 10, 10]),
        "target": torch.randn(4, 10, 1),  # Target is still single dimension
    }

    output = module(batch)

    # Output should have quantile dimensions
    assert output.shape == (4, 10, len(quantiles))
    assert torch.isfinite(output).all()


def test_attention_lstm_module_quantile_training_step() -> None:
    """Test training step with quantile loss."""
    quantiles = [0.1, 0.5, 0.9]
    criterion = QuantileLoss(quantiles=quantiles)

    module = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=16,
        num_layers=1,
        n_heads=2,
        criterion=criterion,
    )

    batch = {
        "encoder_input": torch.randn(2, 15, 3),
        "decoder_input": torch.randn(2, 8, 2),
        "decoder_lengths": torch.tensor([8, 8]),
        "target": torch.randn(2, 8, 1),
    }

    step_output = module.training_step(batch, batch_idx=0)

    assert "loss" in step_output
    assert "output" in step_output
    assert "point_prediction" in step_output

    # Loss should be a scalar
    assert step_output["loss"].dim() == 0
    assert torch.isfinite(step_output["loss"])

    # Output should have quantile dimensions
    assert step_output["output"].shape == (2, 8, len(quantiles))

    # Point prediction should be median (shape loses last dimension)
    assert step_output["point_prediction"].shape == (2, 8)


def test_attention_lstm_module_quantile_point_prediction() -> None:
    """Test point prediction extraction from quantile outputs."""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    criterion = QuantileLoss(quantiles=quantiles)

    module = AttentionLSTM(
        num_past_features=4,
        num_future_features=2,
        hidden_size=20,
        num_layers=1,
        n_heads=4,
        criterion=criterion,
    )

    batch = {
        "encoder_input": torch.randn(3, 12, 4),
        "decoder_input": torch.randn(3, 6, 2),
        "decoder_lengths": torch.tensor([6, 6, 6]),
    }

    # Test predict step (no target required)
    predict_output = module.predict_step(batch, batch_idx=0)

    assert "output" in predict_output
    assert "point_prediction" in predict_output

    # Output should have all quantiles
    assert predict_output["output"].shape == (3, 6, len(quantiles))

    # Point prediction should be the median quantile (0.5)
    median_idx = quantiles.index(0.5)
    expected_point_pred = predict_output["output"][:, :, median_idx]

    assert torch.allclose(predict_output["point_prediction"], expected_point_pred)


def test_attention_lstm_module_quantile_vs_mse() -> None:
    """Test that quantile loss behaves differently from MSE loss."""
    quantiles = [0.1, 0.5, 0.9]

    # Create two identical modules with different losses
    quantile_module = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=16,
        num_layers=1,
        n_heads=2,
        criterion=QuantileLoss(quantiles=quantiles),
    )

    mse_module = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=16,
        num_layers=1,
        n_heads=2,
        output_dim=1,
        criterion=torch.nn.MSELoss(),
    )

    # Different output dimensions
    assert quantile_module.model.output_dim == len(quantiles)
    assert mse_module.model.output_dim == 1

    batch = {
        "encoder_input": torch.randn(2, 10, 3),
        "decoder_input": torch.randn(2, 5, 2),
        "decoder_lengths": torch.tensor([5, 5]),
        "target": torch.randn(2, 5, 1),
    }

    quantile_output = quantile_module(batch)
    mse_output = mse_module(batch)

    # Different output shapes
    assert quantile_output.shape == (2, 5, len(quantiles))
    assert mse_output.shape == (2, 5, 1)


def test_attention_lstm_module_quantile_loss_computation() -> None:
    """Test that quantile loss is computed correctly."""
    quantiles = [0.25, 0.5, 0.75]
    criterion = QuantileLoss(quantiles=quantiles)

    module = AttentionLSTM(
        num_past_features=2,
        num_future_features=1,
        hidden_size=12,
        num_layers=1,
        n_heads=3,
        criterion=criterion,
    )

    batch = {
        "encoder_input": torch.randn(1, 8, 2),
        "decoder_input": torch.randn(1, 4, 1),
        "decoder_lengths": torch.tensor([4]),
        "target": torch.randn(1, 4, 1),
    }

    # Forward pass through model
    module.eval()  # Set to eval mode to disable dropout
    output = module(batch)

    # Manually compute loss
    manual_loss = criterion(output, batch["target"])

    # Loss from training step
    step_output = module.training_step(batch, batch_idx=0)
    training_loss = step_output["loss"]

    # Should be the same
    assert torch.allclose(manual_loss, training_loss, atol=1e-6)


def test_attention_lstm_module_quantile_different_sizes() -> None:
    """Test quantile loss with different quantile configurations."""
    test_cases = [
        [0.5],  # Single median
        [0.1, 0.9],  # Two-sided
        [0.05, 0.25, 0.5, 0.75, 0.95],  # Five quantiles
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Many quantiles
    ]

    for quantiles in test_cases:
        criterion = QuantileLoss(quantiles=quantiles)

        module = AttentionLSTM(
            num_past_features=2,
            num_future_features=1,
            hidden_size=8,
            num_layers=1,
            n_heads=2,
            criterion=criterion,
        )

        assert module.model.output_dim == len(quantiles)

        batch = {
            "encoder_input": torch.randn(1, 5, 2),
            "decoder_input": torch.randn(1, 3, 1),
            "decoder_lengths": torch.tensor([3]),
            "target": torch.randn(1, 3, 1),
        }

        output = module(batch)
        assert output.shape == (1, 3, len(quantiles))

        step_output = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(step_output["loss"])
        assert step_output["point_prediction"].shape == (1, 3)


def test_attention_lstm_module_quantile_variable_lengths() -> None:
    """Test quantile loss with variable sequence lengths."""
    quantiles = [0.1, 0.5, 0.9]
    criterion = QuantileLoss(quantiles=quantiles)

    module = AttentionLSTM(
        num_past_features=3,
        num_future_features=2,
        hidden_size=16,
        num_layers=1,
        n_heads=2,
        criterion=criterion,
    )

    batch = {
        "encoder_input": torch.randn(3, 15, 3),
        "decoder_input": torch.randn(3, 10, 2),
        "decoder_lengths": torch.tensor([8, 10, 6]),  # Variable lengths
        "target": torch.randn(3, 10, 1),
    }

    output = module(batch)
    assert output.shape == (3, 10, len(quantiles))

    step_output = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(step_output["loss"])
    assert step_output["output"].shape == (3, 10, len(quantiles))
    assert step_output["point_prediction"].shape == (3, 10)
