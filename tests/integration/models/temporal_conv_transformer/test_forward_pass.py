from __future__ import annotations

import typing
import warnings

import torch

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.temporal_conv_transformer import (
    TCT,
    TemporalConvTransformer,
    TemporalConvTransformerModel,
)


def test_tct_forward_pass_simple(
    tct_module: TemporalConvTransformer,
    tct_module_config: dict[str, typing.Any],
) -> None:
    """Test simple forward pass with manually created tensors."""
    x_past = torch.rand(
        1,
        tct_module_config["ctxt_seq_len"],
        tct_module_config["num_past_features"],
    )
    x_future = torch.rand(
        1,
        tct_module_config["tgt_seq_len"],
        tct_module_config["num_future_features"],
    )

    batch = {
        "encoder_input": x_past,
        "decoder_input": x_future,
        "encoder_lengths": torch.tensor([[1.0]]),
        "decoder_lengths": torch.tensor([[1.0]]),
        "encoder_mask": torch.ones_like(x_past),
        "decoder_mask": torch.ones_like(x_future),
    }

    with torch.no_grad():
        y = tct_module(batch)["output"]

    assert y.shape[:2] == x_future.shape[:2]


def test_tct_forward_pass_with_datamodule(
    encoder_decoder_datamodule: EncoderDecoderDataModule,
    tct_module: TemporalConvTransformer,
) -> None:
    """Test forward pass with real EncoderDecoderDataModule."""
    # Prepare data module
    encoder_decoder_datamodule.prepare_data()
    encoder_decoder_datamodule.setup()

    # Adjust hyperparameters for TCT requirements
    encoder_decoder_datamodule.hparams["min_ctxt_seq_len"] = (
        400  # Long enough for compression
    )
    encoder_decoder_datamodule.hparams["min_tgt_seq_len"] = (
        100  # Long enough for compression
    )
    encoder_decoder_datamodule.hparams["randomize_seq_len"] = True

    # Hack to remove last 10 values of the val dataset for consistency
    encoder_decoder_datamodule._val_df[0] = encoder_decoder_datamodule._val_df[0].iloc[
        :-10
    ]

    # Test training pipeline
    dataloader = encoder_decoder_datamodule.train_dataloader()
    batch = next(iter(dataloader))

    tct_module.on_train_start()
    tct_module.on_train_epoch_start()

    with torch.no_grad():
        losses = tct_module.training_step(batch, 0)

    for key in ("loss",):
        assert key in losses

    tct_module.on_train_epoch_end()
    tct_module.on_train_end()

    # Test validation pipeline
    dataloader = encoder_decoder_datamodule.val_dataloader()
    batch = next(iter(dataloader))
    last_batch = list(dataloader)[-1]

    tct_module.on_validation_start()
    tct_module.on_validation_epoch_start()

    with torch.no_grad():
        outputs = tct_module.validation_step(batch, 0)
        last_outputs = tct_module.validation_step(last_batch, 0)

    tct_module.on_validation_epoch_end()
    tct_module.on_validation_end()

    for key in (
        "loss",
        "output",
    ):
        assert key in outputs

    for key in (
        "loss",
        "output",
    ):
        assert key in last_outputs
        assert not torch.isnan(last_outputs[key]).any()


def test_tct_training_convergence():
    """Test that TCT can converge on a simple pattern."""
    # Create simple synthetic data with learnable pattern
    batch_size = 4
    encoder_len = 400
    decoder_len = 100
    num_past_features = 8
    num_future_features = 4

    # Create model with reasonable compression
    model = TemporalConvTransformer(
        num_past_features=num_past_features,
        num_future_features=num_future_features,
        output_dim=1,
        hidden_dim=64,
        compression_factor=4,
        dropout=0.0,  # Disable dropout for deterministic training
    )

    # Create synthetic pattern: linear trend + noise
    def create_batch():
        encoder_input = torch.randn(batch_size, encoder_len, num_past_features)
        decoder_input = torch.randn(batch_size, decoder_len, num_future_features)

        # Create simple learnable pattern: sum of first two encoder features
        target = (
            encoder_input[:, -decoder_len:, 0:1] + encoder_input[:, -decoder_len:, 1:2]
        ) * 0.5 + torch.randn(batch_size, decoder_len, 1) * 0.1

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "target": target,
        }

    # Train for a few steps
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    initial_loss = None
    final_loss = None

    for step in range(10):
        batch = create_batch()

        optimizer.zero_grad()
        output = model.training_step(batch, step)
        loss = output["loss"]

        if step == 0:
            initial_loss = loss.item()
        if step == 9:
            final_loss = loss.item()

        loss.backward()
        optimizer.step()

    # Loss should decrease (model should learn something)
    assert final_loss < initial_loss
    assert torch.isfinite(torch.tensor(final_loss))


def test_tct_prediction_consistency():
    """Test that TCT predictions are consistent across runs in eval mode."""
    model = TemporalConvTransformer(
        num_past_features=6,
        num_future_features=3,
        output_dim=1,
        hidden_dim=32,
        compression_factor=2,
        dropout=0.0,
    )
    model.eval()

    # Fixed input for consistency testing
    torch.manual_seed(42)
    batch = {
        "encoder_input": torch.randn(2, 200, 6),
        "decoder_input": torch.randn(2, 50, 3),
    }

    predictions = []
    for _ in range(3):
        with torch.no_grad():
            output = model.predict_step(batch, 0)
            predictions.append(output["output"])

    # All predictions should be identical in eval mode
    for i in range(1, len(predictions)):
        assert torch.allclose(predictions[0], predictions[i], atol=1e-6)


def test_tct_with_different_sequence_lengths():
    """Test TCT with various sequence length combinations."""
    model = TemporalConvTransformer(
        num_past_features=8,
        num_future_features=4,
        output_dim=1,
        hidden_dim=32,
        compression_factor=4,
    )

    test_cases = [
        (400, 100),  # Standard case
        (800, 200),  # Longer sequences
        (600, 150),  # Different ratios
        (400, 50),  # Shorter decoder
    ]

    for encoder_len, decoder_len in test_cases:
        batch = {
            "encoder_input": torch.randn(2, encoder_len, 8),
            "decoder_input": torch.randn(2, decoder_len, 4),
            "target": torch.randn(2, decoder_len, 1),
        }

        # Should handle different lengths gracefully
        output = model.training_step(batch, 0)
        assert output["output"].shape == (2, decoder_len, 1)
        assert torch.isfinite(output["loss"])


def test_tct_attention_weights_analysis():
    """Test that attention weights have expected properties."""
    model = TemporalConvTransformerModel(
        num_past_features=8,
        num_future_features=4,
        output_dim=1,
        hidden_dim=64,
        num_attention_heads=8,
        compression_factor=4,
    )
    model.eval()

    batch = {
        "encoder_input": torch.randn(2, 400, 8),
        "decoder_input": torch.randn(2, 100, 4),
    }

    with torch.no_grad():
        output = model(batch)

    attention_weights = output["attention_weights"]

    # Attention weights should be finite and non-negative
    assert torch.isfinite(attention_weights).all()
    assert (attention_weights >= 0).all()

    # Check attention weight dimensions
    batch_size, compressed_decoder_len, num_heads, total_compressed_len = (
        attention_weights.shape
    )
    assert batch_size == 2
    assert num_heads == 8
    assert compressed_decoder_len == 100 // 4  # 25
    assert total_compressed_len == (400 + 100) // 4  # 125


def test_tct_memory_efficiency_integration():
    """Test TCT memory efficiency with realistic large sequences."""
    model = TemporalConvTransformer(
        num_past_features=16,
        num_future_features=8,
        output_dim=1,
        hidden_dim=128,
        compression_factor=8,  # High compression for efficiency
    )

    # Large sequences that would be prohibitive for standard transformers
    batch = {
        "encoder_input": torch.randn(4, 2000, 16),
        "decoder_input": torch.randn(4, 500, 8),
        "target": torch.randn(4, 500, 1),
    }

    # Should handle large sequences efficiently
    output = model.training_step(batch, 0)
    assert output["output"].shape == (4, 500, 1)
    assert torch.isfinite(output["loss"])


def test_tct_gradient_flow_integration():
    """Test gradient flow through entire TCT pipeline."""
    model = TemporalConvTransformer(
        num_past_features=8,
        num_future_features=4,
        output_dim=1,
        hidden_dim=64,
        compression_factor=4,
    )

    batch = {
        "encoder_input": torch.randn(2, 400, 8, requires_grad=True),
        "decoder_input": torch.randn(2, 100, 4, requires_grad=True),
        "target": torch.randn(2, 100, 1),
    }

    output = model.training_step(batch, 0)
    loss = output["loss"]
    loss.backward()

    # Check that gradients flow to inputs
    assert batch["encoder_input"].grad is not None
    assert batch["decoder_input"].grad is not None
    assert torch.isfinite(batch["encoder_input"].grad).all()
    assert torch.isfinite(batch["decoder_input"].grad).all()

    # Check that all model parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_tct_state_dict_integration():
    """Test state dict save/load integration."""
    model1 = TemporalConvTransformer(
        num_past_features=6,
        num_future_features=3,
        output_dim=1,
        hidden_dim=32,
        compression_factor=2,
    )

    # Get state dict
    state_dict = model1.state_dict()

    # Create new model and load state dict
    model2 = TemporalConvTransformer(
        num_past_features=6,
        num_future_features=3,
        output_dim=1,
        hidden_dim=32,
        compression_factor=2,
    )
    model2.load_state_dict(state_dict)

    # Models should produce identical outputs
    batch = {
        "encoder_input": torch.randn(2, 200, 6),
        "decoder_input": torch.randn(2, 50, 3),
    }

    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(batch)
        output2 = model2(batch)

    assert torch.allclose(output1["output"], output2["output"], atol=1e-6)


def test_tct_device_transfer_integration():
    """Test device transfer integration."""
    model = TemporalConvTransformer(
        num_past_features=6,
        num_future_features=3,
        output_dim=1,
        hidden_dim=32,
        compression_factor=2,
    )

    batch_cpu = {
        "encoder_input": torch.randn(2, 200, 6),
        "decoder_input": torch.randn(2, 50, 3),
        "target": torch.randn(2, 50, 1),
    }

    # Test CPU
    output_cpu = model.training_step(batch_cpu, 0)
    assert output_cpu["output"].device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        batch_cuda = {k: v.cuda() for k, v in batch_cpu.items()}

        output_cuda = model_cuda.training_step(batch_cuda, 0)
        assert output_cuda["output"].device.type == "cuda"
        assert torch.isfinite(output_cuda["loss"])


def test_tct_alias_integration():
    """Test TCT alias integration with full pipeline."""
    model = TCT(
        num_past_features=8,
        num_future_features=4,
        output_dim=1,
        hidden_dim=32,
        compression_factor=2,
    )

    batch = {
        "encoder_input": torch.randn(2, 200, 8),
        "decoder_input": torch.randn(2, 50, 4),
        "target": torch.randn(2, 50, 1),
    }

    # Full training pipeline with alias
    model.on_train_start()
    model.on_train_epoch_start()

    train_output = model.training_step(batch, 0)
    assert torch.isfinite(train_output["loss"])

    model.on_train_epoch_end()

    # Validation pipeline
    model.on_validation_start()
    model.on_validation_epoch_start()

    val_output = model.validation_step(batch, 0)
    assert torch.isfinite(val_output["loss"])

    model.on_validation_epoch_end()
    model.on_validation_end()


def test_tct_sequence_length_warning_integration():
    """Test sequence length warning integration with real data pipeline."""
    # Create model with high compression that will trigger warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = TemporalConvTransformer(
            num_past_features=8,
            num_future_features=4,
            compression_factor=8,
            max_dilation=16,
            hidden_dim=32,
        )

        # Should warn during creation
        assert len(w) >= 1
        assert any("Sequence Requirements" in str(warning.message) for warning in w)

    # Test with short sequences that will trigger runtime warnings
    short_batch = {
        "encoder_input": torch.randn(2, 100, 8),  # Too short for compression_factor=8
        "decoder_input": torch.randn(2, 25, 4),  # Too short
        "target": torch.randn(2, 25, 1),
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        output = model.training_step(short_batch, 0)

        # Should warn during forward pass
        runtime_warnings = [
            warning for warning in w if issubclass(warning.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) >= 1

    # Should still produce valid output despite warnings
    assert torch.isfinite(output["loss"])
    assert output["output"].shape == (2, 25, 1)
