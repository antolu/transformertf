from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from transformertf.data.dataset import EncoderDecoderPredictDataset
from transformertf.data.transform import TransformCollection

CONTEXT_LENGTH = 2
PREDICTION_LENGTH = 2


IN_NAME = "input"
TGT_NAME = "target"


@pytest.fixture
def past_covariates(x_data: np.ndarray, y_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            IN_NAME: x_data[:CONTEXT_LENGTH],
            TGT_NAME: y_data[:CONTEXT_LENGTH],
        }
    )


@pytest.fixture
def future_covariates(x_data: np.ndarray, y_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            IN_NAME: x_data[CONTEXT_LENGTH:],
        }
    )


@pytest.fixture
def past_target(y_data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TGT_NAME: y_data[:CONTEXT_LENGTH],
        }
    )


@pytest.fixture
def future_target(y_data: np.ndarray) -> np.ndarray:
    return y_data[CONTEXT_LENGTH:]


@pytest.fixture
def input_transform() -> TransformCollection:
    return TransformCollection([])


@pytest.fixture
def target_transform() -> TransformCollection:
    return TransformCollection([])


def test_create_predict_dataset_basic(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )


def test_create_predict_dataset_basic_no_transform(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
) -> None:
    EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
    )


def test_create_predict_dataset_no_colnames(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    with pytest.raises(ValueError):
        EncoderDecoderPredictDataset(
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            past_target=past_target,
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            input_transforms={IN_NAME: input_transform},
            target_transform=target_transform,
        )


def test_create_predict_dataset_np(
    x_data: np.ndarray,
    y_data: np.ndarray,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    past_covariates = x_data[:CONTEXT_LENGTH]
    future_covariates = x_data[CONTEXT_LENGTH:]
    past_target = y_data[:CONTEXT_LENGTH]

    EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )


def test_create_predict_dataset_np_no_transform(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    past_covariates = x_data[:CONTEXT_LENGTH]
    future_covariates = x_data[CONTEXT_LENGTH:]
    past_target = y_data[:CONTEXT_LENGTH]

    EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
    )


def test_create_predict_dataset_np_no_colnames(
    x_data: np.ndarray,
    y_data: np.ndarray,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    past_covariates = x_data[:CONTEXT_LENGTH]
    future_covariates = x_data[CONTEXT_LENGTH:]
    past_target = y_data[:CONTEXT_LENGTH]

    EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )


def test_create_predict_dataset_np_long_context(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    past_covariates = x_data
    future_covariates = x_data[CONTEXT_LENGTH:]
    past_target = y_data[:CONTEXT_LENGTH]

    # the dataset should truncate the past covariates to the context length
    EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )


def test_predict_dataset_access_properties(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    assert dataset.input_transform == {IN_NAME: input_transform}
    assert dataset.target_transform == target_transform


def test_predict_dataset_correct_length(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    assert len(dataset) == 4


def test_predict_dataset_append_target_context(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    dataset.append_past_target(past_target[TGT_NAME].values)

    assert len(dataset) == 4


def test_predict_dataset_access_elements(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    _ = dataset[0]

    with pytest.raises(IndexError):
        _ = dataset[1]


def test_predict_dataset_iterate_without_extra_context(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    num_samples = 0
    with pytest.raises(IndexError):
        for _ in dataset:
            num_samples += 1
            pass

    assert num_samples == 1


def test_predict_dataset_iterate_with_extra_context(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    future_target: np.ndarray,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    num_samples = 0
    for i, _ in enumerate(dataset):
        dataset.append_past_target(
            future_target[i * PREDICTION_LENGTH : (i + 1) * PREDICTION_LENGTH]
        )
        num_samples += 1

    assert num_samples == 4


def test_predict_dataset_iterate_with_too_much_extra_context(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    future_target: np.ndarray,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    for i, _ in enumerate(dataset):
        dataset.append_past_target(
            future_target[i * PREDICTION_LENGTH : (i + 1) * PREDICTION_LENGTH]
        )

    with pytest.raises(ValueError):
        dataset.append_past_target(future_target[:1])


def test_predict_dataset_with_dataloader(
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame,
    input_transform: TransformCollection,
    target_transform: TransformCollection,
) -> None:
    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        input_columns=[IN_NAME],
        target_column=TGT_NAME,
        input_transforms={IN_NAME: input_transform},
        target_transform=target_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    for batch in dataloader:
        assert batch["encoder_input"].shape == (1, CONTEXT_LENGTH, 2)
        assert batch["decoder_input"].shape == (1, PREDICTION_LENGTH, 2)
        break
