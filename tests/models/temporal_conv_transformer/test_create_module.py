from __future__ import annotations

import pytest
import torch

from transformertf.models.temporal_conv_transformer import (
    TCT,
    TemporalConvTransformer,
    TemporalConvTransformerModel,
)


class TestTemporalConvTransformerCreation:
    """Test suite for TCT model creation."""

    def test_create_temporal_conv_transformer_model(self):
        """Test creating the core TemporalConvTransformerModel."""
        model = TemporalConvTransformerModel(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,
            hidden_dim=32,
            num_attention_heads=4,
            compression_factor=2,
        )

        assert model is not None
        assert isinstance(model, TemporalConvTransformerModel)
        assert model.num_past_features == 10
        assert model.num_future_features == 5
        assert model.output_dim == 1
        assert model.hidden_dim == 32
        assert model.compression_factor == 2

    def test_create_temporal_conv_transformer_lightning(self):
        """Test creating the Lightning TemporalConvTransformer module."""
        # Use custom criterion to avoid QuantileLoss auto-adjustment
        custom_criterion = torch.nn.MSELoss()
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=3,
            output_dim=2,
            hidden_dim=64,
            compression_factor=4,
            criterion=custom_criterion,
        )

        assert model is not None
        assert isinstance(model, TemporalConvTransformer)
        assert model.hparams["num_past_features"] == 8
        assert model.hparams["num_future_features"] == 3
        assert model.hparams["output_dim"] == 2
        assert model.hparams["hidden_dim"] == 64
        assert model.hparams["compression_factor"] == 4

    def test_tct_alias(self):
        """Test that TCT alias works correctly."""
        model = TCT(
            num_past_features=5,
            num_future_features=2,
            output_dim=1,
        )

        assert model is not None
        assert isinstance(model, TemporalConvTransformer)
        assert model.hparams["num_past_features"] == 5
        assert model.hparams["num_future_features"] == 2

    def test_default_parameters(self):
        """Test model creation with default parameters."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
        )

        # Check default values
        assert model.hparams["output_dim"] == 1
        assert model.hparams["hidden_dim"] == 256
        assert model.hparams["compression_factor"] == 4
        assert hasattr(model, "model")
        assert hasattr(model, "criterion")

    def test_custom_parameters(self):
        """Test model creation with custom parameters."""
        # Use custom criterion to avoid QuantileLoss auto-adjustment
        custom_criterion = torch.nn.MSELoss()
        model = TemporalConvTransformer(
            num_past_features=15,
            num_future_features=8,
            output_dim=3,
            hidden_dim=128,
            num_attention_heads=8,
            compression_factor=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=0.2,
            activation="gelu",
            max_dilation=16,
            criterion=custom_criterion,
        )

        assert model.hparams["num_past_features"] == 15
        assert model.hparams["num_future_features"] == 8
        assert model.hparams["output_dim"] == 3
        assert model.hparams["hidden_dim"] == 128
        assert model.hparams["compression_factor"] == 8
        assert model.hparams["max_dilation"] == 16

    def test_quantile_loss_with_multiple_outputs(self):
        """Test that QuantileLoss is used for multiple output dimensions."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            output_dim=3,  # Multiple outputs should trigger QuantileLoss
            hidden_dim=32,
        )

        # Should have QuantileLoss for multiple outputs
        from transformertf.nn import QuantileLoss

        assert isinstance(model.criterion, QuantileLoss) or (
            hasattr(model.criterion, "_orig_mod")
            and isinstance(model.criterion._orig_mod, QuantileLoss)
        )

    def test_mse_loss_with_single_output(self):
        """Test that MSELoss is used for single output dimension."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            output_dim=1,  # Single output should trigger MSELoss
            hidden_dim=32,
        )

        # Should have MSELoss for single output
        assert isinstance(model.criterion, torch.nn.MSELoss) or (
            hasattr(model.criterion, "_orig_mod")
            and isinstance(model.criterion._orig_mod, torch.nn.MSELoss)
        )

    def test_custom_criterion(self):
        """Test model creation with custom criterion."""
        custom_criterion = torch.nn.L1Loss()

        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            criterion=custom_criterion,
            hidden_dim=32,
        )

        assert model.criterion is custom_criterion

    @pytest.mark.parametrize("compression_factor", [2, 4, 8, 16])
    def test_different_compression_factors(self, compression_factor):
        """Test model creation with different compression factors."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            compression_factor=compression_factor,
            hidden_dim=32,
        )

        assert model.hparams["compression_factor"] == compression_factor
        assert model.model.compression_factor == compression_factor

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_attention_heads(self, num_heads):
        """Test model creation with different numbers of attention heads."""
        # Ensure hidden_dim is divisible by num_heads
        hidden_dim = 32 if num_heads <= 4 else 64

        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            num_attention_heads=num_heads,
            hidden_dim=hidden_dim,
        )

        assert model.model.attention.num_heads == num_heads

    def test_model_has_required_components(self):
        """Test that model has all required components."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            hidden_dim=32,
        )

        # Check that model has required components
        assert hasattr(model.model, "past_encoder")
        assert hasattr(model.model, "future_encoder")
        assert hasattr(model.model, "attention")
        assert hasattr(model.model, "_decoder_params")

    def test_model_parameter_consistency(self):
        """Test that parameters are consistent between Lightning and core model."""
        # Use output_dim=1 to avoid QuantileLoss adjustment
        lightning_model = TemporalConvTransformer(
            num_past_features=12,
            num_future_features=6,
            output_dim=1,
            hidden_dim=64,
            compression_factor=4,
        )

        core_model = lightning_model.model

        assert core_model.num_past_features == 12
        assert core_model.num_future_features == 6
        assert core_model.output_dim == 1
        assert core_model.hidden_dim == 64
        assert core_model.compression_factor == 4

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid compression factor
        with pytest.raises((ValueError, TypeError)):
            TemporalConvTransformer(
                num_past_features=10,
                num_future_features=5,
                compression_factor=0,  # Invalid
            )

        # Test invalid hidden_dim with num_heads
        with pytest.raises((ValueError, AssertionError)):
            TemporalConvTransformer(
                num_past_features=10,
                num_future_features=5,
                hidden_dim=17,  # Not divisible by num_heads=8
                num_attention_heads=8,
            )

    def test_model_device_placement(self):
        """Test that model can be moved to different devices."""
        model = TemporalConvTransformer(
            num_past_features=5,
            num_future_features=3,
            hidden_dim=32,
        )

        # Test CPU placement (default)
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA placement if available
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == "cuda"

    def test_model_eval_train_modes(self):
        """Test that model can switch between train and eval modes."""
        model = TemporalConvTransformer(
            num_past_features=5,
            num_future_features=3,
            hidden_dim=32,
        )

        # Test train mode (default)
        assert model.training

        # Test eval mode
        model.eval()
        assert not model.training

        # Test back to train mode
        model.train()
        assert model.training

    def test_model_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = TemporalConvTransformer(
            num_past_features=10,
            num_future_features=5,
            hidden_dim=64,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert total_params > 1000  # At least some parameters
        assert total_params < 10000000  # Not too many for test model

    def test_model_state_dict(self):
        """Test that model state_dict works correctly."""
        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            hidden_dim=32,
        )

        state_dict = model.state_dict()

        # Should have state dict entries
        assert len(state_dict) > 0
        assert all(isinstance(v, torch.Tensor) for v in state_dict.values())

    def test_model_reproducibility(self):
        """Test that model creation is reproducible with same seed."""
        torch.manual_seed(42)
        model1 = TemporalConvTransformer(
            num_past_features=6,
            num_future_features=3,
            hidden_dim=32,
        )

        torch.manual_seed(42)
        model2 = TemporalConvTransformer(
            num_past_features=6,
            num_future_features=3,
            hidden_dim=32,
        )

        # Models should have same initial parameters
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters(), strict=False
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-6)

    @pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
    def test_different_activations(self, activation):
        """Test model creation with different activation functions."""
        model = TemporalConvTransformer(
            num_past_features=5,
            num_future_features=3,
            activation=activation,
            hidden_dim=32,
        )

        assert model is not None
        # Activation is properly set in components
