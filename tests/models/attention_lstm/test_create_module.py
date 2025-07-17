from __future__ import annotations

from transformertf.models.attention_lstm import AttentionLSTM, AttentionLSTMModel


def test_create_attention_lstm() -> None:
    """Test basic instantiation of AttentionLSTM model."""
    model = AttentionLSTMModel(
        num_past_features=5,
        num_future_features=3,
        hidden_size=64,
        num_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    assert model is not None
    assert model.num_past_features == 5
    assert model.num_future_features == 3
    assert model.hidden_size == 64
    assert model.num_layers == 2
    assert model.n_heads == 4
    assert model.dropout == 0.1
    assert model.use_gating is True


def test_create_attention_lstm_module() -> None:
    """Test basic instantiation of AttentionLSTM Lightning wrapper."""
    model = AttentionLSTM(
        num_past_features=5,
        num_future_features=3,
        hidden_size=64,
        num_layers=2,
        n_heads=4,
        dropout=0.1,
        use_gating=True,
    )

    assert model is not None
    assert model.hparams.num_past_features == 5
    assert model.hparams.num_future_features == 3
    assert model.hparams.hidden_size == 64
    assert model.hparams.num_layers == 2
    assert model.hparams.n_heads == 4
    assert model.hparams.dropout == 0.1
    assert model.hparams.use_gating is True


def test_create_attention_lstm_without_gating() -> None:
    """Test AttentionLSTM model without gating mechanism."""
    model = AttentionLSTMModel(
        num_past_features=4,
        num_future_features=2,
        hidden_size=32,
        num_layers=1,
        n_heads=2,
        dropout=0.0,
        use_gating=False,
    )

    assert model is not None
    assert model.use_gating is False
