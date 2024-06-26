from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

from .._dtype import convert_data
from .._sample_generator import (
    EncoderDecoderSample,
    TransformerPredictionSampleGenerator,
)
from ..transform import BaseTransform, TransformType
from ._base import AbstractTimeSeriesDataset, DataSetType


class EncoderDecoderPredictDataset(
    AbstractTimeSeriesDataset, torch.utils.data.IterableDataset
):
    _dataset_type = DataSetType.PREDICT

    def __init__(
        self,
        past_covariates: pd.DataFrame | np.ndarray,
        future_covariates: pd.DataFrame | np.ndarray,
        past_target: pd.DataFrame | np.ndarray | pd.Series,
        context_length: int,
        prediction_length: int,
        input_transforms: dict[str, BaseTransform] | None = None,
        target_transform: BaseTransform | None = None,
        input_columns: list[str] | None = None,
        target_column: str | None = None,
        known_past_columns: list[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        A dataset for predicting future time steps using an encoder-decoder
        style model, since the model must predict the future time steps
        autoregressively if the desired future time steps are longer than
        the target length.

        This effectively means that the model will predict the future time
        steps one by one, using the previously predicted time steps as
        context for the next prediction. This dataset will generate the
        context and target windows for each prediction step.

        The dataset requires to use the `update_transforms` method to update

        Parameters
        ----------
        past_covariates : pd.DataFrame | np.ndarray
            Past covariates to be used for prediction. If a DataFrame, the
            `input_columns` parameter must be provided. The data should not
            be preprocessed with the transforms, as the transforms will be
            applied during iteration. If a numpy array, the data must be
            of the same order during training.
        future_covariates : pd.DataFrame | np.ndarray
            Future covariates to be used for prediction. If a DataFrame, the
            `input_columns` parameter must be provided. The data should not
            be preprocessed with the transforms, as the transforms will be
            applied during iteration. The samples generated by this dataset
            will predict as many future time steps as there are rows in this
            data. If a numpy array, the data must be of the same order during
            training.
        past_target : pd.DataFrame | np.ndarray | pd.Series
            Past target to be used for prediction. If a DataFrame, the
            `target_column` parameter must be provided. The data should not
            be preprocessed with the transforms, as the transforms will be
            applied during iteration.
        context_length : int
            Length of the context window to be used for prediction.
        prediction_length : int
            Length of the target window to be used for prediction.
        input_transforms : dict[str, BaseTransform], optional
            A dictionary with the input column names as keys and the
            transforms to be applied to each column as values. If None,
            no transforms will be applied. The transforms will be applied
            during iteration, by default None.
        target_transform : BaseTransform, optional
            A transform to be applied to the target column. If None, no
            transform will be applied. The transform will be applied during
            iteration, by default None.
        input_columns : list[str], optional
            A list with the column names of the past covariates. If
            `past_covariates` is a DataFrame, this parameter must be
            provided.
        target_column : str, optional
            The name of the target column. If `past_target` is a DataFrame,
            this parameter must be provided.
        """
        torch.utils.data.IterableDataset.__init__(self)
        AbstractTimeSeriesDataset.__init__(self)

        self._input_transform = input_transforms or {}
        self._target_transform = target_transform

        self._context_length = context_length
        self._target_length = prediction_length
        self._dtype = dtype

        past_known_covariates = (
            extract_covariates_from_df(past_covariates, known_past_columns)
            if known_past_columns is not None
            and isinstance(past_covariates, pd.DataFrame)
            else None
        )

        past_covariates = extract_covariates_from_df(past_covariates, input_columns)
        future_covariates = extract_covariates_from_df(future_covariates, input_columns)
        past_target = extract_covariates_from_df(past_target, target_column)

        past_covariates = keep_only_context(past_covariates, context_length)
        past_target = keep_only_context(past_target, context_length)
        past_known_covariates = (
            keep_only_context(past_known_covariates, context_length)  # type: ignore[arg-type]
            if past_known_covariates is not None
            else None
        )

        self._past_covariates = apply_transforms(
            past_covariates,
            {k: v for k, v in self._input_transform.items() if k in input_columns},
        )
        self._future_covariates = apply_transforms(
            future_covariates,
            {k: v for k, v in self._input_transform.items() if k in input_columns},
        )
        self._past_target = apply_transforms(
            past_target, self._target_transform, past_covariates[..., 0]
        )
        past_known_transforms = {
            k: v for k, v in self._input_transform.items() if k in known_past_columns
        }
        self._past_known_covariates = (
            apply_transforms(past_known_covariates, past_known_transforms)  # type: ignore[arg-type]
            if past_known_covariates is not None
            else None
        )

        # convert to torch tensors
        self._past_covariates = convert_data(self._past_covariates, dtype)[0]
        self._future_covariates = convert_data(self._future_covariates, dtype)[0]
        self._past_target = convert_data(self._past_target, dtype)[0]
        self._past_known_covariates = (
            convert_data(self._past_known_covariates, dtype)[0]
            if self._past_known_covariates is not None
            else None
        )

        self._sample_generator = TransformerPredictionSampleGenerator(
            past_covariates=self._past_covariates,
            future_covariates=self._future_covariates,
            past_targets=self._past_target,
            context_length=context_length,
            prediction_length=prediction_length,
            known_past_covariates=self._past_known_covariates,
        )

    def append_past_target(
        self,
        past_target: np.ndarray | torch.Tensor,
        *,
        transform: bool = False,
    ) -> None:
        """
        Appends the past target to the dataset. This method must be called
        between iterations to append the past target to the dataset.
        """
        if transform:
            past_target = apply_transforms(past_target, self._target_transform)
        else:
            past_target = torch.as_tensor(past_target)

        self._sample_generator.add_target_context(past_target)

    def append_past_covariates(
        self,
        past_covariates: np.ndarray | torch.Tensor,
        *,
        transform: bool = False,
    ) -> None:
        """
        Appends the past covariates to the dataset. This method must be called
        between iterations to append the past covariates to the dataset.
        """
        if transform:
            past_covariates = apply_transforms(past_covariates, self._input_transform)
        else:
            past_covariates = torch.as_tensor(past_covariates)

        self._sample_generator.add_known_past_context(past_covariates)

    def __getitem__(self, idx: int) -> EncoderDecoderSample:
        return self._sample_generator[idx]

    def __iter__(self) -> typing.Generator[EncoderDecoderSample, None, None]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(self._sample_generator)


def extract_covariates_from_df(
    data: pd.DataFrame | np.ndarray | pd.Series,
    columns: list[str] | str | None = None,
) -> np.ndarray:
    if isinstance(data, pd.Series):
        return data.to_numpy()
    if isinstance(data, pd.DataFrame):
        if columns is None:
            msg = "The columns parameter must be provided if the data is a DataFrame."
            raise ValueError(msg)
        return data[columns].to_numpy()
    return data  # type: ignore[return-value]


def keep_only_context(
    data: np.ndarray,
    context_length: int,
) -> np.ndarray:
    if len(data) < context_length:
        msg = "The data is shorter than the context length."
        raise ValueError(msg)
    return data[-context_length:]


def apply_transforms(
    data: np.ndarray | pd.Series,
    transforms: dict[str, BaseTransform] | BaseTransform | None = None,
    dependency: np.ndarray | pd.Series | None = None,
) -> torch.Tensor:
    """
    Applies the provided transforms to the data. If the data is a 2D array,
    the transforms must be a dictionary with the column names as keys and
    the transforms as values. If the data is a 1D array or a pandas Series,
    the transforms must be a single transform.

    Parameters
    ----------
    data : np.ndarray | pd.Series
        The data to be transformed.
    transforms : dict[str, BaseTransform] | BaseTransform
        The transforms to be applied to the data. If the data is a 2D array,
        the transforms must be a dictionary with the column names as keys and
        the transforms as values. If the data is a 1D array or a pandas Series,
        the transforms must be a single transform.

    Returns
    -------
    torch.Tensor
        The transformed data.
    """
    if transforms is None or (isinstance(transforms, dict) and len(transforms) == 0):
        return torch.as_tensor(data)
    assert transforms is not None  # mypy

    if isinstance(data, pd.Series):
        data = data.to_numpy()

    if data.ndim == 1:
        if not isinstance(transforms, BaseTransform) and (
            isinstance(transforms, dict) and len(transforms) > 1
        ):
            msg = "More than one transform was provided for a single column."
            raise ValueError(msg)
        if isinstance(transforms, dict):
            transforms = next(iter(transforms.values()))
        assert isinstance(transforms, BaseTransform)  # mypy

        if dependency is not None and transforms.transform_type == TransformType.XY:
            dependency = (
                dependency.to_numpy()
                if isinstance(dependency, pd.Series)
                else dependency
            )

            return transforms.transform(dependency, data)
        return transforms.transform(data)

    # If the data is a 2D array
    num_cols = data.shape[1]

    if isinstance(transforms, BaseTransform):
        msg = "A single transform was provided for multiple columns."
        raise ValueError(msg)  # noqa: TRY004
    if (
        isinstance(transforms, dict)
        and num_cols != len(transforms)
        and len(transforms) > 0
    ):
        msg = (
            "The number of transforms must match the number of columns "
            "if transforms are provided."
        )
        raise ValueError(msg)

    if len(transforms) == 0:
        return torch.as_tensor(data)

    output = []
    for col, (_col_name, transform) in enumerate(transforms.items()):
        output.append(transform.transform(data[:, col]))

    return torch.stack(output, dim=1)
