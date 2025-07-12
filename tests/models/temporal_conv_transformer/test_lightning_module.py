from __future__ import annotations

import pytest
import torch

from transformertf.models.temporal_conv_transformer import (
    TCT,
    TemporalConvTransformer,
)
from transformertf.nn import QuantileLoss


@pytest.fixture(scope="module")
def sample_batch():
    """Sample batch for Lightning module testing."""
    return {
        "encoder_input": torch.randn(4, 400, 8),
        "decoder_input": torch.randn(4, 100, 4),
        "encoder_lengths": torch.full((4, 1), 400),
        "decoder_lengths": torch.full((4, 1), 100),
        "target": torch.randn(4, 100, 1),
    }


class TestTemporalConvTransformerLightning:
    """Test suite for TCT Lightning module functionality."""

    def test_lightning_module_creation(self):
        """Test basic Lightning module creation."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=64,
        )

        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "criterion")
        assert hasattr(model, "hparams")

    def test_training_step(self, sample_batch):
        """Test training step functionality."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
        )

        output = model.training_step(sample_batch, 0)

        assert isinstance(output, dict)
        assert "loss" in output
        assert "output" in output
        assert "point_prediction" in output

        # Check shapes
        assert output["output"].shape == (4, 100, 1)
        assert output["point_prediction"].shape == (4, 100, 1)
        assert torch.isfinite(output["loss"])

    def test_validation_step(self, sample_batch):
        """Test validation step functionality."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
        )

        output = model.validation_step(sample_batch, 0)

        assert isinstance(output, dict)
        assert "loss" in output
        assert "output" in output
        assert "point_prediction" in output

        # Check shapes
        assert output["output"].shape == (4, 100, 1)
        assert output["point_prediction"].shape == (4, 100, 1)
        assert torch.isfinite(output["loss"])

    def test_predict_step(self):
        """Test prediction step functionality."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compression_factor=4,
        )

        # Prediction batch without target
        predict_batch = {
            "encoder_input": torch.randn(2, 400, 8),
            "decoder_input": torch.randn(2, 100, 4),
        }

        output = model.predict_step(predict_batch, 0)

        assert isinstance(output, dict)
        assert "output" in output
        assert "point_prediction" in output
        assert "attention_weights" in output

        # Check shapes
        assert output["output"].shape == (2, 100, 1)
        assert output["point_prediction"].shape == (2, 100, 1)

    def test_loss_computation_single_output(self, sample_batch):
        """Test loss computation with single output dimension."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,  # Single output should use MSELoss
            hidden_dim=32,
        )

        # Should use MSELoss for single output
        assert isinstance(model.criterion, torch.nn.MSELoss) or (
            hasattr(model.criterion, "_orig_mod")
            and isinstance(model.criterion._orig_mod, torch.nn.MSELoss)
        )

        output = model.training_step(sample_batch, 0)
        assert torch.isfinite(output["loss"])

    def test_loss_computation_multiple_outputs(self):
        """Test loss computation with multiple output dimensions."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=3,  # Multiple outputs should use QuantileLoss
            hidden_dim=32,
        )

        # Should use QuantileLoss for multiple outputs
        assert isinstance(model.criterion, QuantileLoss) or (
            hasattr(model.criterion, "_orig_mod")
            and isinstance(model.criterion._orig_mod, QuantileLoss)
        )

        batch = {
            "encoder_input": torch.randn(2, 400, 8),
            "decoder_input": torch.randn(2, 100, 4),
            "target": torch.randn(2, 100, 1),  # Target always has 1 dimension
        }

        output = model.training_step(batch, 0)
        assert torch.isfinite(output["loss"])

    def test_custom_criterion(self, sample_batch):
        """Test model with custom criterion."""
        custom_criterion = torch.nn.L1Loss()

        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            criterion=custom_criterion,
            hidden_dim=32,
        )

        assert model.criterion is custom_criterion

        output = model.training_step(sample_batch, 0)
        assert torch.isfinite(output["loss"])

    def test_point_prediction_extraction(self, sample_batch):
        """Test point prediction extraction from model output."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
        )

        output = model.training_step(sample_batch, 0)

        # For single output, point_prediction should equal output
        assert torch.allclose(output["point_prediction"], output["output"], atol=1e-6)

    def test_point_prediction_extraction_quantile(self):
        """Test point prediction extraction from quantile output."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=3,  # Multiple outputs for quantile
            hidden_dim=32,
        )

        batch = {
            "encoder_input": torch.randn(2, 400, 8),
            "decoder_input": torch.randn(2, 100, 4),
            "target": torch.randn(2, 100, 1),
        }

        output = model.training_step(batch, 0)

        # Point prediction should be extracted from quantile output
        assert output["point_prediction"].shape == (2, 100, 1)
        assert not torch.allclose(
            output["point_prediction"], output["output"], atol=1e-6
        )  # Should be different due to quantile extraction

    def test_model_state_transitions(self, sample_batch):
        """Test model behavior in different states (train/eval)."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            dropout=0.1,  # Enable dropout to see difference
        )

        # Train mode
        model.train()
        output_train1 = model.training_step(sample_batch, 0)
        output_train2 = model.training_step(sample_batch, 0)

        # Outputs should be different in train mode (due to dropout)
        assert not torch.allclose(
            output_train1["output"], output_train2["output"], atol=1e-6
        )

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval1 = model.validation_step(sample_batch, 0)
            output_eval2 = model.validation_step(sample_batch, 0)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1["output"], output_eval2["output"], atol=1e-6)

    def test_hyperparameter_storage(self):
        """Test that hyperparameters are properly stored."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=6,
            output_dim=1,  # Use 1 to avoid QuantileLoss adjustment
            hidden_dim=128,
            compression_factor=8,
            dropout=0.2,
        )

        hparams = model.hparams

        assert hparams["num_past_features"] == 10
        assert hparams["num_future_features"] == 6
        assert hparams["output_dim"] == 1
        assert hparams["hidden_dim"] == 128
        assert hparams["compression_factor"] == 8
        assert hparams["dropout"] == 0.2

    def test_tct_alias_lightning_functionality(self, sample_batch):
        """Test that TCT alias works for Lightning functionality."""
        model = TCT(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
        )

        # Test training step
        output = model.training_step(sample_batch, 0)
        assert isinstance(output, dict)
        assert "loss" in output
        assert torch.isfinite(output["loss"])

    def test_compile_model_functionality(self, sample_batch):
        """Test model compilation functionality."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            output_dim=1,
            hidden_dim=32,
            compile_model=True,
        )

        # Simulate fit start to trigger compilation
        model.on_fit_start()

        # Should still work after compilation
        output = model.training_step(sample_batch, 0)
        assert torch.isfinite(output["loss"])

    def test_logging_metrics_configuration(self, sample_batch):
        """Test different logging metrics configurations."""
        # Default metrics
        model1 = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
        )

        # Custom metrics
        model2 = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
            logging_metrics=["MSE", "MAE"],
        )

        # No additional metrics
        model3 = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
            logging_metrics=[],
        )

        # All should work
        for model in [model1, model2, model3]:
            output = model.training_step(sample_batch, 0)
            assert torch.isfinite(output["loss"])

    def test_device_placement_lightning(self):
        """Test device placement for Lightning module."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
        )

        # Test CPU placement (default)
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA placement if available
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == "cuda"

            # Test with CUDA batch
            batch = {
                "encoder_input": torch.randn(2, 400, 8, device="cuda"),
                "decoder_input": torch.randn(2, 100, 4, device="cuda"),
                "target": torch.randn(2, 100, 1, device="cuda"),
            }

            output = model.training_step(batch, 0)
            assert output["loss"].device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_parameter_freezing(self, sample_batch):
        """Test parameter freezing functionality."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
            trainable_parameters=["model.attention"],  # Only train attention
        )

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())

        # Should have fewer trainable parameters
        assert trainable_params < total_params

    def test_state_dict_compilation_handling(self):
        """Test state dict handling with compilation."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
            compile_model=True,
        )

        # Get state dict before compilation
        state_dict_before = model.state_dict()

        # Trigger compilation
        model.on_fit_start()

        # Get state dict after compilation
        state_dict_after = model.state_dict()

        # Should have same keys (compilation effects should be removed)
        assert set(state_dict_before.keys()) == set(state_dict_after.keys())

    def test_different_batch_configurations(self):
        """Test Lightning module with different batch configurations."""
        model = TemporalConvTransformer(
            num_past_features=6,
            num_future_features=3,
            hidden_dim=32,
            compression_factor=2,
        )

        test_configs = [
            # (batch_size, encoder_len, decoder_len)
            (1, 200, 50),
            (4, 400, 100),
            (8, 600, 150),
        ]

        for batch_size, encoder_len, decoder_len in test_configs:
            batch = {
                "encoder_input": torch.randn(batch_size, encoder_len, 6),
                "decoder_input": torch.randn(batch_size, decoder_len, 3),
                "target": torch.randn(batch_size, decoder_len, 1),
            }

            output = model.training_step(batch, 0)
            assert output["output"].shape == (batch_size, decoder_len, 1)
            assert torch.isfinite(output["loss"])

    def test_validation_outputs_collection(self, sample_batch):
        """Test that validation outputs are properly collected."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
        )

        # Simulate validation epoch start
        model.on_validation_epoch_start()

        # Run validation steps
        model.validation_step(sample_batch, 0, dataloader_idx=0)
        model.validation_step(sample_batch, 1, dataloader_idx=0)

        # Check that outputs are collected
        validation_outputs = model.validation_outputs
        assert 0 in validation_outputs  # dataloader_idx=0
        assert len(validation_outputs[0]) == 2  # Two validation steps

    def test_gradient_accumulation_compatibility(self, sample_batch):
        """Test compatibility with gradient accumulation."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
        )

        # Simulate multiple training steps without optimizer step
        outputs = []
        for i in range(3):
            output = model.training_step(sample_batch, i)
            outputs.append(output)

        # All should produce valid losses
        for output in outputs:
            assert torch.isfinite(output["loss"])
            assert output["loss"].requires_grad

    def test_error_handling_invalid_batch(self):
        """Test error handling with invalid batch structure."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
        )

        # Invalid batch missing required keys
        invalid_batch = {
            "encoder_input": torch.randn(2, 400, 8),
            # Missing decoder_input
            "target": torch.randn(2, 100, 1),
        }

        with pytest.raises((KeyError, ValueError)):
            model.training_step(invalid_batch, 0)

    def test_memory_efficiency_lightning(self):
        """Test memory efficiency in Lightning module."""
        model = TemporalConvTransformer(
            num_past_features=16,
            num_future_features=8,
            hidden_dim=64,
            compression_factor=8,  # High compression for efficiency
        )

        # Large batch
        batch = {
            "encoder_input": torch.randn(8, 1000, 16),
            "decoder_input": torch.randn(8, 250, 8),
            "target": torch.randn(8, 250, 1),
        }

        output = model.training_step(batch, 0)
        assert output["output"].shape == (8, 250, 1)
        assert torch.isfinite(output["loss"])
