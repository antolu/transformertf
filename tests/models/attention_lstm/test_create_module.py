from __future__ import annotations

from transformertf.models.attention_lstm import AttentionLSTM, AttentionLSTMModel


def test_create_attention_lstm() -> None:
    """Test basic instantiation of AttentionLSTM model."""
    model = AttentionLSTMModel(
        num_past_features=5,
        num_future_features=3,
        d_model=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    assert model is not None
    assert model.num_past_features == 5
    assert model.num_future_features == 3
    assert model.d_model == 64
    assert model.num_layers == 2
    assert model.num_heads == 4
    assert model.dropout == 0.1
    assert model.use_gating is True


def test_create_attention_lstm_module() -> None:
    """Test basic instantiation of AttentionLSTM Lightning wrapper."""
    model = AttentionLSTM(
        num_past_features=5,
        num_future_features=3,
        d_model=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    assert model is not None
    assert model.hparams.num_past_features == 5
    assert model.hparams.num_future_features == 3
    assert model.hparams.d_model == 64
    assert model.hparams.num_layers == 2
    assert model.hparams.num_heads == 4
    assert model.hparams.dropout == 0.1
    assert model.hparams.use_gating is True


def test_create_attention_lstm_without_gating() -> None:
    """Test AttentionLSTM model without gating mechanism."""
    model = AttentionLSTMModel(
        num_past_features=4,
        num_future_features=2,
        d_model=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_gating=False,
    )

    assert model is not None
    assert model.use_gating is False
