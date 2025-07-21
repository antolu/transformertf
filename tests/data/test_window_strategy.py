"""
Tests for WindowStrategy pattern implementation.

This module tests the strategy pattern for window generation,
including validation, parameter handling, and sample generator creation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transformertf.data._window_strategy import (
    PredictWindowStrategy,
    RandomizedWindowStrategy,
    TimeSeriesWindowStrategy,
    TransformerWindowStrategy,
    WindowStrategy,
    WindowStrategyFactory,
)


def test_timeseries_window_strategy_init_basic() -> None:
    """Test basic initialization."""
    strategy = TimeSeriesWindowStrategy(seq_len=10, stride=2)

    assert strategy.seq_len == 10
    assert strategy.stride == 2
    assert strategy.min_seq_len is None
    assert strategy.randomize_seq_len is False
    assert strategy.predict is False


def test_timeseries_window_strategy_init_with_randomization() -> None:
    """Test initialization with randomization parameters."""
    strategy = TimeSeriesWindowStrategy(
        seq_len=10,
        stride=1,
        min_seq_len=5,
        randomize_seq_len=True,
    )

    assert strategy.seq_len == 10
    assert strategy.min_seq_len == 5
    assert strategy.randomize_seq_len is True


def test_timeseries_window_strategy_validate_parameters_valid() -> None:
    """Test parameter validation with valid parameters."""
    # Should not raise any exception
    TimeSeriesWindowStrategy(seq_len=10, stride=1)
    TimeSeriesWindowStrategy(
        seq_len=10,
        stride=1,
        min_seq_len=5,
        randomize_seq_len=True,
    )


def test_timeseries_window_strategy_validate_parameters_invalid_seq_len() -> None:
    """Test parameter validation with invalid seq_len."""
    with pytest.raises(ValueError, match="seq_len must be positive"):
        TimeSeriesWindowStrategy(seq_len=0, stride=1)

    with pytest.raises(ValueError, match="seq_len must be positive"):
        TimeSeriesWindowStrategy(seq_len=-1, stride=1)


def test_timeseries_window_strategy_validate_parameters_invalid_stride() -> None:
    """Test parameter validation with invalid stride."""
    with pytest.raises(ValueError, match="stride must be positive"):
        TimeSeriesWindowStrategy(seq_len=10, stride=0)

    with pytest.raises(ValueError, match="stride must be positive"):
        TimeSeriesWindowStrategy(seq_len=10, stride=-1)


def test_timeseries_window_strategy_validate_parameters_randomization_without_min_seq_len() -> (
    None
):
    """Test parameter validation with randomization but no min_seq_len."""
    with pytest.raises(ValueError, match="min_seq_len must be specified"):
        TimeSeriesWindowStrategy(
            seq_len=10,
            stride=1,
            randomize_seq_len=True,
        )


def test_timeseries_window_strategy_validate_parameters_invalid_min_seq_len() -> None:
    """Test parameter validation with invalid min_seq_len."""
    with pytest.raises(ValueError, match="min_seq_len must be positive"):
        TimeSeriesWindowStrategy(
            seq_len=10,
            stride=1,
            min_seq_len=0,
            randomize_seq_len=True,
        )

    with pytest.raises(
        ValueError, match="min_seq_len must be less than or equal to seq_len"
    ):
        TimeSeriesWindowStrategy(
            seq_len=10,
            stride=1,
            min_seq_len=15,
            randomize_seq_len=True,
        )


def test_timeseries_window_strategy_create_sample_generators_basic() -> None:
    """Test creating sample generators with basic parameters."""
    strategy = TimeSeriesWindowStrategy(seq_len=5, stride=1)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [pd.DataFrame({"target": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})]

    generators = strategy.create_sample_generators(input_data, target_data)

    assert len(generators) == 1
    assert len(generators[0]) > 0  # Should have some samples


def test_timeseries_window_strategy_create_sample_generators_with_stride() -> None:
    """Test creating sample generators with stride."""
    strategy = TimeSeriesWindowStrategy(seq_len=3, stride=2)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8]})]
    target_data = [pd.DataFrame({"target": [10, 20, 30, 40, 50, 60, 70, 80]})]

    generators = strategy.create_sample_generators(input_data, target_data)

    # Should have stride number of generators
    assert len(generators) == 2


def test_timeseries_window_strategy_create_sample_generators_predict_mode() -> None:
    """Test creating sample generators in predict mode."""
    strategy = TimeSeriesWindowStrategy(seq_len=5, stride=1, predict=True)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [None]  # No target in predict mode

    generators = strategy.create_sample_generators(input_data, target_data)

    # Should create generators successfully for predict mode
    assert len(generators) == 1


def test_timeseries_window_strategy_get_window_parameters() -> None:
    """Test getting window parameters."""
    strategy = TimeSeriesWindowStrategy(
        seq_len=10,
        stride=2,
        min_seq_len=5,
        randomize_seq_len=True,
        predict=False,
    )

    params = strategy.get_window_parameters()

    assert params["seq_len"] == 10
    assert params["stride"] == 2
    assert params["min_seq_len"] == 5
    assert params["randomize_seq_len"] is True
    assert params["predict"] is False


def test_transformer_window_strategy_init_basic() -> None:
    """Test basic initialization."""
    strategy = TransformerWindowStrategy(ctx_seq_len=10, tgt_seq_len=5, stride=1)

    assert strategy.ctx_seq_len == 10
    assert strategy.tgt_seq_len == 5
    assert strategy.stride == 1
    assert strategy.min_ctx_seq_len is None
    assert strategy.min_tgt_seq_len is None
    assert strategy.randomize_seq_len is False
    assert strategy.predict is False


def test_transformer_window_strategy_init_with_randomization() -> None:
    """Test initialization with randomization parameters."""
    strategy = TransformerWindowStrategy(
        ctx_seq_len=10,
        tgt_seq_len=5,
        stride=1,
        min_ctx_seq_len=5,
        min_tgt_seq_len=2,
        randomize_seq_len=True,
    )

    assert strategy.ctx_seq_len == 10
    assert strategy.tgt_seq_len == 5
    assert strategy.min_ctx_seq_len == 5
    assert strategy.min_tgt_seq_len == 2
    assert strategy.randomize_seq_len is True


def test_transformer_window_strategy_validate_parameters_valid() -> None:
    """Test parameter validation with valid parameters."""
    # Should not raise any exception
    TransformerWindowStrategy(ctx_seq_len=10, tgt_seq_len=5, stride=1)
    TransformerWindowStrategy(
        ctx_seq_len=10,
        tgt_seq_len=5,
        stride=1,
        min_ctx_seq_len=5,
        min_tgt_seq_len=2,
        randomize_seq_len=True,
    )


def test_transformer_window_strategy_validate_parameters_invalid_ctx_seq_len() -> None:
    """Test parameter validation with invalid ctx_seq_len."""
    with pytest.raises(ValueError, match="ctx_seq_len must be positive"):
        TransformerWindowStrategy(ctx_seq_len=0, tgt_seq_len=5, stride=1)


def test_transformer_window_strategy_validate_parameters_invalid_tgt_seq_len() -> None:
    """Test parameter validation with invalid tgt_seq_len."""
    with pytest.raises(ValueError, match="tgt_seq_len must be positive"):
        TransformerWindowStrategy(ctx_seq_len=10, tgt_seq_len=0, stride=1)


def test_transformer_window_strategy_validate_parameters_randomization_without_min_lengths() -> (
    None
):
    """Test parameter validation with randomization but no min lengths."""
    with pytest.raises(ValueError, match="min_ctx_seq_len must be specified"):
        TransformerWindowStrategy(
            ctx_seq_len=10,
            tgt_seq_len=5,
            stride=1,
            randomize_seq_len=True,
        )


def test_transformer_window_strategy_validate_parameters_invalid_min_ctx_seq_len() -> (
    None
):
    """Test parameter validation with invalid min_ctx_seq_len."""
    with pytest.raises(ValueError, match="min_ctx_seq_len.*must be positive"):
        TransformerWindowStrategy(
            ctx_seq_len=10,
            tgt_seq_len=5,
            stride=1,
            min_ctx_seq_len=0,
            min_tgt_seq_len=2,
            randomize_seq_len=True,
        )

    with pytest.raises(
        ValueError,
        match="min_ctx_seq_len must be less than or equal to ctx_seq_len",
    ):
        TransformerWindowStrategy(
            ctx_seq_len=10,
            tgt_seq_len=5,
            stride=1,
            min_ctx_seq_len=15,
            min_tgt_seq_len=2,
            randomize_seq_len=True,
        )


def test_transformer_window_strategy_create_sample_generators_basic() -> None:
    """Test creating sample generators with basic parameters."""
    strategy = TransformerWindowStrategy(ctx_seq_len=5, tgt_seq_len=3, stride=1)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [pd.DataFrame({"target": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})]

    generators = strategy.create_sample_generators(input_data, target_data)

    assert len(generators) == 1
    assert len(generators[0]) > 0  # Should have some samples


def test_transformer_window_strategy_create_sample_generators_with_optional_data() -> (
    None
):
    """Test creating sample generators with optional known_past_data and time_data."""
    strategy = TransformerWindowStrategy(ctx_seq_len=5, tgt_seq_len=3, stride=1)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [pd.DataFrame({"target": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})]
    known_past_data = [pd.DataFrame({"past_feature": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})]
    time_data = [pd.DataFrame({"time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})]

    generators = strategy.create_sample_generators(
        input_data,
        target_data,
        known_past_data=known_past_data,
        time_data=time_data,
    )

    assert len(generators) == 1


def test_transformer_window_strategy_get_window_parameters() -> None:
    """Test getting window parameters."""
    strategy = TransformerWindowStrategy(
        ctx_seq_len=10,
        tgt_seq_len=5,
        stride=2,
        min_ctx_seq_len=5,
        min_tgt_seq_len=2,
        randomize_seq_len=True,
        predict=False,
    )

    params = strategy.get_window_parameters()

    assert params["ctx_seq_len"] == 10
    assert params["tgt_seq_len"] == 5
    assert params["stride"] == 2
    assert params["min_ctx_seq_len"] == 5
    assert params["min_tgt_seq_len"] == 2
    assert params["randomize_seq_len"] is True
    assert params["predict"] is False


def test_predict_window_strategy_init_basic() -> None:
    """Test basic initialization."""
    strategy = PredictWindowStrategy(window_size=10, stride=2)

    assert strategy.window_size == 10
    assert strategy.stride == 2
    assert strategy.zero_pad is True


def test_predict_window_strategy_init_no_zero_pad() -> None:
    """Test initialization without zero padding."""
    strategy = PredictWindowStrategy(window_size=10, stride=2, zero_pad=False)

    assert strategy.zero_pad is False


def test_predict_window_strategy_validate_parameters_valid() -> None:
    """Test parameter validation with valid parameters."""
    # Should not raise any exception
    PredictWindowStrategy(window_size=10, stride=1)
    PredictWindowStrategy(window_size=10, stride=2, zero_pad=False)


def test_predict_window_strategy_validate_parameters_invalid_window_size() -> None:
    """Test parameter validation with invalid window_size."""
    with pytest.raises(ValueError, match="window_size must be positive"):
        PredictWindowStrategy(window_size=0, stride=1)

    with pytest.raises(ValueError, match="window_size must be positive"):
        PredictWindowStrategy(window_size=-1, stride=1)


def test_predict_window_strategy_validate_parameters_invalid_stride() -> None:
    """Test parameter validation with invalid stride."""
    with pytest.raises(ValueError, match="stride must be positive"):
        PredictWindowStrategy(window_size=10, stride=0)

    with pytest.raises(ValueError, match="stride must be positive"):
        PredictWindowStrategy(window_size=10, stride=-1)


def test_predict_window_strategy_create_sample_generators_basic() -> None:
    """Test creating sample generators with basic parameters."""
    strategy = PredictWindowStrategy(window_size=5, stride=1)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [None]  # No target in predict mode

    generators = strategy.create_sample_generators(input_data, target_data)

    assert len(generators) == 1
    assert generators[0]._window_generator._window_size == 5
    assert generators[0]._window_generator._stride == 1


def test_predict_window_strategy_get_window_parameters() -> None:
    """Test getting window parameters."""
    strategy = PredictWindowStrategy(window_size=10, stride=2, zero_pad=False)

    params = strategy.get_window_parameters()

    assert params["window_size"] == 10
    assert params["stride"] == 2
    assert params["zero_pad"] is False


def test_randomized_window_strategy_init_basic() -> None:
    """Test basic initialization."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = RandomizedWindowStrategy(base_strategy)

    assert strategy.base_strategy is base_strategy
    assert strategy.randomize_training is True


def test_randomized_window_strategy_init_no_randomize() -> None:
    """Test initialization without randomization."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = RandomizedWindowStrategy(base_strategy, randomize_training=False)

    assert strategy.randomize_training is False


def test_randomized_window_strategy_validate_parameters_delegates_to_base() -> None:
    """Test that parameter validation delegates to base strategy."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = RandomizedWindowStrategy(base_strategy)

    # Should not raise any exception
    strategy.validate_parameters()


def test_randomized_window_strategy_create_sample_generators_adds_randomization() -> (
    None
):
    """Test that randomization is added to kwargs during training."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = RandomizedWindowStrategy(base_strategy, randomize_training=True)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [pd.DataFrame({"target": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})]

    # Mock the base strategy's create_sample_generators to check kwargs
    called_kwargs = {}

    def mock_create_generators(input_data, target_data, **kwargs):
        called_kwargs.update(kwargs)
        return []

    base_strategy.create_sample_generators = mock_create_generators

    strategy.create_sample_generators(input_data, target_data, predict=False)

    assert called_kwargs["randomize_seq_len"] is True


def test_randomized_window_strategy_create_sample_generators_no_randomization_in_predict() -> (
    None
):
    """Test that randomization is not added during prediction."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = RandomizedWindowStrategy(base_strategy, randomize_training=True)

    input_data = [pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})]
    target_data = [None]

    # Mock the base strategy's create_sample_generators to check kwargs
    called_kwargs = {}

    def mock_create_generators(input_data, target_data, **kwargs):
        called_kwargs.update(kwargs)
        return []

    base_strategy.create_sample_generators = mock_create_generators

    strategy.create_sample_generators(input_data, target_data, predict=True)

    assert "randomize_seq_len" not in called_kwargs


def test_randomized_window_strategy_get_window_parameters_includes_randomization() -> (
    None
):
    """Test that window parameters include randomization info."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = RandomizedWindowStrategy(base_strategy, randomize_training=True)

    params = strategy.get_window_parameters()

    assert params["seq_len"] == 10
    assert params["stride"] == 1
    assert params["randomize_training"] is True


def test_window_strategy_factory_create_timeseries_strategy() -> None:
    """Test creating time series strategy."""
    strategy = WindowStrategyFactory.create_timeseries_strategy(
        seq_len=10,
        stride=2,
        min_seq_len=5,
        randomize_seq_len=True,
        predict=False,
    )

    assert isinstance(strategy, TimeSeriesWindowStrategy)
    assert strategy.seq_len == 10
    assert strategy.stride == 2
    assert strategy.min_seq_len == 5
    assert strategy.randomize_seq_len is True
    assert strategy.predict is False


def test_window_strategy_factory_create_transformer_strategy() -> None:
    """Test creating transformer strategy."""
    strategy = WindowStrategyFactory.create_transformer_strategy(
        ctx_seq_len=10,
        tgt_seq_len=5,
        stride=2,
        min_ctx_seq_len=5,
        min_tgt_seq_len=2,
        randomize_seq_len=True,
        predict=False,
    )

    assert isinstance(strategy, TransformerWindowStrategy)
    assert strategy.ctx_seq_len == 10
    assert strategy.tgt_seq_len == 5
    assert strategy.stride == 2
    assert strategy.min_ctx_seq_len == 5
    assert strategy.min_tgt_seq_len == 2
    assert strategy.randomize_seq_len is True
    assert strategy.predict is False


def test_window_strategy_factory_create_predict_strategy() -> None:
    """Test creating predict strategy."""
    strategy = WindowStrategyFactory.create_predict_strategy(
        window_size=10,
        stride=2,
        zero_pad=False,
    )

    assert isinstance(strategy, PredictWindowStrategy)
    assert strategy.window_size == 10
    assert strategy.stride == 2
    assert strategy.zero_pad is False


def test_window_strategy_factory_create_randomized_strategy() -> None:
    """Test creating randomized strategy."""
    base_strategy = TimeSeriesWindowStrategy(seq_len=10, stride=1)
    strategy = WindowStrategyFactory.create_randomized_strategy(
        base_strategy=base_strategy,
        randomize_training=True,
    )

    assert isinstance(strategy, RandomizedWindowStrategy)
    assert strategy.base_strategy is base_strategy
    assert strategy.randomize_training is True


def test_window_strategy_interface_compliance() -> None:
    """Test that all strategies implement the WindowStrategy interface."""
    strategies = [
        TimeSeriesWindowStrategy(seq_len=10, stride=1),
        TransformerWindowStrategy(ctx_seq_len=10, tgt_seq_len=5, stride=1),
        PredictWindowStrategy(window_size=10, stride=1),
        RandomizedWindowStrategy(
            TimeSeriesWindowStrategy(seq_len=10, stride=1),
            randomize_training=True,
        ),
    ]

    for strategy in strategies:
        assert isinstance(strategy, WindowStrategy)

        # All strategies should have these methods
        assert hasattr(strategy, "create_sample_generators")
        assert hasattr(strategy, "get_window_parameters")
        assert hasattr(strategy, "validate_parameters")

        # Methods should be callable
        assert callable(strategy.create_sample_generators)
        assert callable(strategy.get_window_parameters)
        assert callable(strategy.validate_parameters)


def test_window_strategy_composition() -> None:
    """Test that strategies can be composed (e.g., with RandomizedWindowStrategy)."""
    # Create a base strategy
    base_strategy = TransformerWindowStrategy(ctx_seq_len=10, tgt_seq_len=5, stride=1)

    # Wrap it with randomization
    randomized_strategy = RandomizedWindowStrategy(
        base_strategy, randomize_training=True
    )

    # Should maintain the same interface
    assert isinstance(randomized_strategy, WindowStrategy)

    # Should be able to get parameters
    params = randomized_strategy.get_window_parameters()
    assert "ctx_seq_len" in params
    assert "tgt_seq_len" in params
    assert "randomize_training" in params


def test_window_strategy_factory_creates_correct_types() -> None:
    """Test that factory creates the correct strategy types."""
    timeseries_strategy = WindowStrategyFactory.create_timeseries_strategy(seq_len=10)
    transformer_strategy = WindowStrategyFactory.create_transformer_strategy(
        ctx_seq_len=10, tgt_seq_len=5
    )
    predict_strategy = WindowStrategyFactory.create_predict_strategy(window_size=10)

    assert isinstance(timeseries_strategy, TimeSeriesWindowStrategy)
    assert isinstance(transformer_strategy, TransformerWindowStrategy)
    assert isinstance(predict_strategy, PredictWindowStrategy)

    # All should be WindowStrategy instances
    assert isinstance(timeseries_strategy, WindowStrategy)
    assert isinstance(transformer_strategy, WindowStrategy)
    assert isinstance(predict_strategy, WindowStrategy)


def test_window_strategy_parameter_validation_across_strategies() -> None:
    """Test that parameter validation works consistently across strategies."""
    # Test that invalid parameters raise appropriate errors
    invalid_cases = [
        (
            lambda: TimeSeriesWindowStrategy(seq_len=0, stride=1),
            "seq_len must be positive",
        ),
        (
            lambda: TransformerWindowStrategy(ctx_seq_len=0, tgt_seq_len=5, stride=1),
            "ctx_seq_len must be positive",
        ),
        (
            lambda: PredictWindowStrategy(window_size=0, stride=1),
            "window_size must be positive",
        ),
        (
            lambda: WindowStrategyFactory.create_timeseries_strategy(seq_len=-1),
            "seq_len must be positive",
        ),
    ]

    for strategy_factory, expected_error in invalid_cases:
        with pytest.raises(ValueError, match=expected_error):
            strategy_factory()
