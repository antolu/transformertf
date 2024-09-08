"""
Tests for the `transformertf.data.TransformerPredictionSampleGenerator` class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformertf.data import TransformerPredictionSampleGenerator

CONTEXT_LENGTH = 2
PREDICTION_LENGTH = 2


@pytest.fixture(scope="module")
def past_covariates(x_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": x_data[:CONTEXT_LENGTH]})


@pytest.fixture(scope="module")
def future_covariates(x_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": x_data[CONTEXT_LENGTH:]})


@pytest.fixture(scope="module")
def past_targets(y_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"y": y_data[:CONTEXT_LENGTH]})


@pytest.fixture(scope="module")
def future_targets(y_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"y": y_data[CONTEXT_LENGTH:]})


def test_create_prediction_sample_generator_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )


def test_create_prediction_sample_generator_wrong_context_length_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TransformerPredictionSampleGenerator(
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            past_targets=past_targets,
            context_length=CONTEXT_LENGTH + 1,
            prediction_length=PREDICTION_LENGTH,
        )


def test_create_prediction_sample_generator_wrong_past_covariate_length_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TransformerPredictionSampleGenerator(
            past_covariates=past_covariates.iloc[:-1],
            future_covariates=future_covariates,
            past_targets=past_targets,
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
        )


def test_create_prediction_sample_generator_wrong_past_target_length_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TransformerPredictionSampleGenerator(
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            past_targets=past_targets.iloc[:-1],
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
        )


def test_prediction_sample_generator_num_samples_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    assert len(pg) == 4


def test_prediction_sample_generator_first_sample_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    sample = pg[0]

    assert "encoder_input" in sample
    assert "decoder_input" in sample

    assert sample["encoder_input"].shape == (CONTEXT_LENGTH, 2)
    assert sample["decoder_input"].shape == (PREDICTION_LENGTH, 2)


def test_prediction_sample_generator_next_sample_raises_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    with pytest.raises(IndexError):
        _ = pg[1]


def test_prediction_sample_generator_add_context_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
    future_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    _ = pg[0]
    pg.add_target_context(future_targets[:PREDICTION_LENGTH])

    assert len(pg) == 4

    sample = pg[1]
    assert "encoder_input" in sample
    assert "decoder_input" in sample

    assert sample["encoder_input"].shape == (CONTEXT_LENGTH, 2)
    assert sample["decoder_input"].shape == (PREDICTION_LENGTH, 2)

    with pytest.raises(IndexError):
        _ = pg[2]


def test_prediction_sample_generator_add_full_context_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
    future_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    _ = pg[0]
    pg.add_target_context(future_targets)

    assert len(pg) == 4

    for i in range(4):
        sample = pg[i]
        assert "encoder_input" in sample
        assert "decoder_input" in sample

        assert sample["encoder_input"].shape == (CONTEXT_LENGTH, 2)
        assert sample["decoder_input"].shape == (PREDICTION_LENGTH, 2)

    with pytest.raises(IndexError):
        _ = pg[4]


def test_prediction_sample_generator_add_full_context_iteratively_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
    future_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    for i in range(4):
        sample = pg[i]
        assert "encoder_input" in sample
        assert "decoder_input" in sample

        assert sample["encoder_input"].shape == (CONTEXT_LENGTH, 2)
        assert sample["decoder_input"].shape == (PREDICTION_LENGTH, 2)

        if i > 0:
            assert np.all(
                sample["encoder_input"]["y"].to_numpy()
                == future_targets[(i - 1) * PREDICTION_LENGTH : i * PREDICTION_LENGTH]
                .to_numpy()
                .flatten()
            )

        pg.add_target_context(
            future_targets[i * PREDICTION_LENGTH : (i + 1) * PREDICTION_LENGTH]
        )

    with pytest.raises(IndexError):
        _ = pg[4]


def test_prediction_sample_generator_add_too_much_context_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
    future_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    _ = pg[0]

    for i in range(4):
        pg.add_target_context(
            future_targets[i * PREDICTION_LENGTH : (i + 1) * PREDICTION_LENGTH]
        )

    with pytest.raises(ValueError):  # noqa: PT011
        pg.add_target_context(future_targets[:1])


def test_prediction_sample_generator_last_sample_zero_padded_1d(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_targets: pd.DataFrame,
    future_targets: pd.DataFrame,
) -> None:
    pg = TransformerPredictionSampleGenerator(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_targets=past_targets,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    _ = pg[0]

    pg.add_target_context(future_targets)

    sample = pg[-1]
    assert "encoder_input" in sample
    assert "decoder_input" in sample

    assert sample["encoder_input"].shape == (CONTEXT_LENGTH, 2)
    assert sample["decoder_input"].shape == (PREDICTION_LENGTH, 2)

    assert np.all(
        sample["encoder_input"]["x"].to_numpy()
        == future_covariates[(3 - 1) * PREDICTION_LENGTH : 3 * PREDICTION_LENGTH]
        .to_numpy()
        .flatten()
    )
    assert np.all(sample["decoder_input"]["x"].to_numpy().flatten() == [9, 0])
    assert np.all(sample["decoder_input"]["y"].to_numpy().flatten() == 0.0)
