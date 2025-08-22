from __future__ import annotations

import pytest
import torch

from transformertf.models.attention_lstm import AttentionLSTM, AttentionLSTMModel

BATCH_SIZE = 4
PAST_SEQ_LEN = 50
FUTURE_SEQ_LEN = 10
NUM_PAST_FEATURES = 5
NUM_FUTURE_FEATURES = 3


@pytest.fixture(scope="module")
def past_covariates() -> torch.Tensor:
    """Create sample past covariates tensor."""
    return torch.rand(BATCH_SIZE, PAST_SEQ_LEN, NUM_PAST_FEATURES)


@pytest.fixture(scope="module")
def future_covariates() -> torch.Tensor:
    """Create sample future covariates tensor."""
    return torch.rand(BATCH_SIZE, FUTURE_SEQ_LEN, NUM_FUTURE_FEATURES)


@pytest.fixture(scope="module")
def encoder_lengths() -> torch.Tensor:
    """Create sample encoder lengths tensor."""
    return torch.tensor([PAST_SEQ_LEN] * BATCH_SIZE)


@pytest.fixture(scope="module")
def decoder_lengths() -> torch.Tensor:
    """Create sample decoder lengths tensor."""
    return torch.tensor([FUTURE_SEQ_LEN] * BATCH_SIZE)


class TestDifferentEncoderDecoderDimensions:
    """Test functionality with different encoder/decoder dimensions."""

    def test_different_dimensions_forward_pass(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        encoder_lengths: torch.Tensor,
        decoder_lengths: torch.Tensor,
    ) -> None:
        """Test forward pass with different encoder/decoder dimensions."""
        d_encoder = 128
        d_decoder = 64

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            output_dim=1,
        )

        # Test forward pass
        output = model(
            past_sequence=past_covariates,
            future_sequence=future_covariates,
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
        )

        assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)
        assert not torch.isnan(output).any()

    def test_encoder_decoder_dimensions_with_return_states(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test forward pass with return_encoder_states=True."""
        d_encoder = 96
        d_decoder = 48

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            num_layers=3,
            num_heads=6,
            output_dim=2,
        )

        output, encoder_states = model(
            past_sequence=past_covariates,
            future_sequence=future_covariates,
            return_encoder_states=True,
        )

        assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 2)
        assert isinstance(encoder_states, tuple)
        assert len(encoder_states) == 2  # (h, c)

        h, c = encoder_states
        # Encoder states should be in encoder dimension
        assert h.shape == (3, BATCH_SIZE, d_encoder)  # (num_layers, batch, d_encoder)
        assert c.shape == (3, BATCH_SIZE, d_encoder)

    def test_state_projection_layers_created(self) -> None:
        """Test that state projection layers are created when dimensions differ."""
        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=128,
            d_decoder=64,
            num_layers=2,
            num_heads=4,
        )

        # Check that projection layers exist
        assert model.state_projection is not None
        assert model.encoder_output_projection is not None
        assert "h_proj" in model.state_projection
        assert "c_proj" in model.state_projection

        # Check projection layer dimensions
        assert model.state_projection["h_proj"].in_features == 128  # d_encoder
        assert model.state_projection["h_proj"].out_features == 64  # d_decoder
        assert model.encoder_output_projection.in_features == 128  # d_encoder
        assert model.encoder_output_projection.out_features == 64  # d_decoder

    def test_no_projection_layers_when_dimensions_equal(self) -> None:
        """Test that no projection layers are created when dimensions are equal."""
        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=64,
            d_decoder=64,
            num_layers=2,
            num_heads=4,
        )

        # Check that projection layers don't exist
        assert model.state_projection is None
        assert model.encoder_output_projection is None

    def test_attention_operates_in_decoder_dimension(self) -> None:
        """Test that attention mechanism uses decoder dimension."""
        d_encoder = 128
        d_decoder = 80
        num_heads = 5

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            num_layers=2,
            num_heads=num_heads,
        )

        # Attention should use decoder dimension
        assert model.attention.d_model == d_decoder
        assert model.attention.num_heads == num_heads

    def test_output_layers_use_decoder_dimension(self) -> None:
        """Test that output-related layers use decoder dimension."""
        d_encoder = 96
        d_decoder = 48
        output_dim = 3

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            use_gating=True,
            output_dim=output_dim,
        )

        # Check gating layer uses decoder dimension
        assert model.gate_add_norm.input_dim == d_decoder

        # Check output head uses decoder dimension
        assert model.output_head.in_features == d_decoder
        assert model.output_head.out_features == output_dim

    def test_layer_norm_uses_decoder_dimension_when_no_gating(self) -> None:
        """Test that layer norm uses decoder dimension when gating is disabled."""
        d_encoder = 72
        d_decoder = 36

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            use_gating=False,
        )

        assert model.gate_add_norm is None
        assert model.layer_norm is not None
        assert model.layer_norm.normalized_shape == (d_decoder,)


class TestBackwardCompatibility:
    """Test backward compatibility with existing d_model parameter."""

    def test_d_model_fallback_when_no_specific_dimensions(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test that d_model is used when d_encoder/d_decoder are not specified."""
        d_model = 64

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_model=d_model,
            num_layers=2,
            num_heads=4,
        )

        # Both encoder and decoder should use d_model
        assert model.d_encoder == d_model
        assert model.d_decoder == d_model
        assert model.encoder.hidden_size == d_model
        assert model.decoder.hidden_size == d_model

        # No projection layers should exist
        assert model.state_projection is None
        assert model.encoder_output_projection is None

        # Forward pass should work
        output = model(past_covariates, future_covariates)
        assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)

    def test_d_encoder_takes_precedence_over_d_model(self) -> None:
        """Test that d_encoder overrides d_model when specified."""
        d_model = 64
        d_encoder = 96

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_model=d_model,
            d_encoder=d_encoder,
            num_layers=2,
            num_heads=4,
        )

        assert model.d_encoder == d_encoder
        assert model.d_decoder == d_model  # Should fallback to d_model
        assert model.encoder.hidden_size == d_encoder
        assert model.decoder.hidden_size == d_model

    def test_d_decoder_takes_precedence_over_d_model(self) -> None:
        """Test that d_decoder overrides d_model when specified."""
        d_model = 64
        d_decoder = 48

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_model=d_model,
            d_decoder=d_decoder,
            num_layers=2,
            num_heads=6,
        )

        assert model.d_encoder == d_model  # Should fallback to d_model
        assert model.d_decoder == d_decoder
        assert model.encoder.hidden_size == d_model
        assert model.decoder.hidden_size == d_decoder

    def test_both_dimensions_override_d_model(self) -> None:
        """Test that both d_encoder and d_decoder override d_model when specified."""
        d_model = 64
        d_encoder = 128
        d_decoder = 32

        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_model=d_model,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            num_layers=2,
            num_heads=4,
        )

        assert model.d_encoder == d_encoder
        assert model.d_decoder == d_decoder
        assert model.encoder.hidden_size == d_encoder
        assert model.decoder.hidden_size == d_decoder


class TestValidation:
    """Test parameter validation for new dimensions."""

    def test_decoder_dimension_divisible_by_num_heads(self) -> None:
        """Test that d_decoder must be divisible by num_heads."""
        with pytest.raises(
            AssertionError, match="d_decoder.*must be divisible by num_heads"
        ):
            AttentionLSTMModel(
                num_past_features=NUM_PAST_FEATURES,
                num_future_features=NUM_FUTURE_FEATURES,
                d_encoder=128,
                d_decoder=65,  # Not divisible by 8
                num_heads=8,
            )

    def test_d_model_divisibility_when_no_d_decoder(self) -> None:
        """Test that d_model divisibility is checked when d_decoder is not specified."""
        with pytest.raises(
            AssertionError, match="d_decoder.*must be divisible by num_heads"
        ):
            AttentionLSTMModel(
                num_past_features=NUM_PAST_FEATURES,
                num_future_features=NUM_FUTURE_FEATURES,
                d_model=65,  # Not divisible by 8, used as d_decoder
                num_heads=8,
            )

    def test_valid_decoder_dimension_divisible_by_num_heads(self) -> None:
        """Test that valid d_decoder divisible by num_heads works."""
        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=127,  # Can be any value
            d_decoder=64,  # Divisible by 8
            num_heads=8,
        )

        assert model.d_encoder == 127
        assert model.d_decoder == 64


class TestLightningModuleIntegration:
    """Test integration with Lightning module."""

    def test_lightning_module_with_different_dimensions(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test Lightning module with different encoder/decoder dimensions."""
        from transformertf.data import EncoderDecoderTargetSample

        d_encoder = 96
        d_decoder = 48

        lightning_module = AttentionLSTM(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=d_encoder,
            d_decoder=d_decoder,
            num_layers=2,
            num_heads=6,
            output_dim=2,
        )

        # Check underlying model has correct dimensions
        assert lightning_module.model.d_encoder == d_encoder
        assert lightning_module.model.d_decoder == d_decoder

        # Test forward pass
        batch: EncoderDecoderTargetSample = {
            "encoder_input": past_covariates,
            "decoder_input": future_covariates,
            "target": torch.rand(BATCH_SIZE, FUTURE_SEQ_LEN, 2),
        }

        result = lightning_module.forward(batch)

        assert isinstance(result, dict)
        assert "output" in result
        assert "encoder_states" in result
        assert result["output"].shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 2)

    def test_lightning_module_backward_compatibility(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test Lightning module backward compatibility with d_model."""
        from transformertf.data import EncoderDecoderTargetSample

        d_model = 64

        lightning_module = AttentionLSTM(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_model=d_model,
            num_layers=2,
            num_heads=4,
        )

        # Check underlying model uses d_model for both dimensions
        assert lightning_module.model.d_encoder == d_model
        assert lightning_module.model.d_decoder == d_model

        # Test forward pass
        batch: EncoderDecoderTargetSample = {
            "encoder_input": past_covariates,
            "decoder_input": future_covariates,
        }

        result = lightning_module.forward(batch)
        assert isinstance(result, dict)
        assert "output" in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_minimal_dimensions(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test with minimal dimensions."""
        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=4,
            d_decoder=2,
            num_layers=1,
            num_heads=1,
            dropout=0.0,
        )

        output = model(past_covariates, future_covariates)
        assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)

    def test_large_dimension_difference(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test with large difference between encoder and decoder dimensions."""
        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=512,
            d_decoder=32,
            num_layers=2,
            num_heads=4,
        )

        output = model(past_covariates, future_covariates)
        assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)
        assert not torch.isnan(output).any()

    def test_decoder_larger_than_encoder(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> None:
        """Test when decoder dimension is larger than encoder."""
        model = AttentionLSTMModel(
            num_past_features=NUM_PAST_FEATURES,
            num_future_features=NUM_FUTURE_FEATURES,
            d_encoder=32,
            d_decoder=128,
            num_layers=2,
            num_heads=8,
        )

        output = model(past_covariates, future_covariates)
        assert output.shape == (BATCH_SIZE, FUTURE_SEQ_LEN, 1)
        assert not torch.isnan(output).any()
