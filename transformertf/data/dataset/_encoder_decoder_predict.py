from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from .._sample_generator import (
    EncoderDecoderSample,
    TransformerPredictionSampleGenerator,
)
from ..transform import BaseTransform
from ._base import AbstractTimeSeriesDataset, DataSetType
from ._encoder_decoder import EncoderDecoderDataset, apply_transforms, convert_sample


class EncoderDecoderPredictDataset(
    AbstractTimeSeriesDataset, torch.utils.data.IterableDataset
):
    _dataset_type = DataSetType.PREDICT

    def __init__(
        self,
        past_covariates: pd.DataFrame,
        future_covariates: pd.DataFrame,
        past_target: pd.DataFrame | np.ndarray | pd.Series,
        context_length: int,
        prediction_length: int,
        input_columns: list[str],
        target_column: str | None = None,
        known_past_columns: list[str] | None = None,
        transforms: dict[str, BaseTransform] | None = None,
        time_format: typing.Literal["relative", "absolute"] = "relative",
        *,
        apply_transforms: bool = True,
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
        past_covariates : pd.DataFrame
            Past covariates to be used for prediction. If a DataFrame, the
            `input_columns` parameter must be provided. The data should not
            be preprocessed with the transforms, as the transforms will be
            applied during iteration. If a numpy array, the data must be
            of the same order during training.
        future_covariates : pd.DataFrame
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
        transforms : dict[str, BaseTransform], optional
            A dictionary with the input column names as keys and the
            transforms to be applied to each column as values. If None,
            no transforms will be applied. The transforms will be applied
            during iteration, by default None.
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

        if isinstance(past_target, pd.DataFrame) and target_column is None:
            msg = (
                "The target_column parameter must be provided if "
                "the past_target is a DataFrame."
            )
            raise ValueError(msg)

        self._transforms = transforms or {}

        self._context_length = context_length
        self._target_length = prediction_length
        self._dtype = dtype
        self._input_columns = input_columns
        self._target_column = target_column
        self._known_past_columns = known_past_columns
        self._time_format = time_format

        self._past_known_covariates = (
            extract_covariates_from_df(past_covariates, known_past_columns).reset_index(
                drop=True
            )
            if known_past_columns is not None
            and isinstance(past_covariates, pd.DataFrame)
            else None
        )

        self._past_covariates: pd.DataFrame = extract_covariates_from_df(
            past_covariates, input_columns
        ).reset_index(drop=True)
        self._future_covariates: pd.DataFrame = extract_covariates_from_df(
            future_covariates, input_columns
        ).reset_index(drop=True)
        self._past_target: npt.NDArray[np.float64] = (
            past_target[target_column].to_numpy()
            if isinstance(past_target, pd.DataFrame)
            else np.array(past_target)
        )

        self._past_covariates = keep_only_context(self._past_covariates, context_length)
        self._past_target = keep_only_context(self._past_target, context_length)
        self._past_known_covariates = (
            keep_only_context(self._past_known_covariates, context_length)  # type: ignore[arg-type]
            if self._past_known_covariates is not None
            else None
        )

        if apply_transforms:
            past_covariate_transforms = {
                k: v
                for k, v in self._transforms.items()
                if k in input_columns and k != "__time__"
            }

            if known_past_columns is not None:
                past_known_covariate_transforms = {
                    k: v for k, v in self._transforms.items() if k in known_past_columns
                }
            else:
                past_known_covariate_transforms = {}

            # assume first feature is what the target depends on
            columns = self._past_covariates.columns
            first_feature = next(iter([col for col in columns if col != "__time__"]))

            self._past_covariates = _apply_transforms(
                self._past_covariates,
                past_covariate_transforms,
            )
            self._future_covariates = _apply_transforms(
                self._future_covariates,
                past_covariate_transforms,
            )
            if target_column in self._transforms:
                self._past_target = _apply_transforms(
                    self._past_target,
                    self._transforms[target_column],
                    self._past_covariates[first_feature].to_numpy(),
                )
            self._past_known_covariates = (
                _apply_transforms(
                    self._past_known_covariates, past_known_covariate_transforms
                )  # type: ignore[arg-type]
                if self._past_known_covariates is not None
                else None
            )

        self._sample_generator = TransformerPredictionSampleGenerator(
            past_covariates=self._past_covariates,
            future_covariates=self._future_covariates,
            past_targets=pd.DataFrame({target_column: self._past_target}),
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
        assert self._target_column is not None
        if isinstance(past_target, np.ndarray):
            past_target = pd.DataFrame({self._target_column: past_target})
        if transform:
            past_target = _apply_transforms(
                past_target, self._transforms[self._target_column]
            )

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
        colname = (self._known_past_columns or [])[0]
        past_covariates = pd.DataFrame({colname: past_covariates})
        if transform:
            colname = (self._known_past_columns or [])[0]
            past_covariates = _apply_transforms(
                past_covariates, {colname: self._transforms[colname]}
            )

        self._sample_generator.add_known_past_context(past_covariates)

    def __getitem__(self, idx: int) -> EncoderDecoderSample:
        sample = self._sample_generator[idx]
        sample = EncoderDecoderDataset._format_time_data(  # noqa: SLF001
            sample, time_format=self._time_format
        )
        sample = apply_transforms(
            sample,  # type: ignore[arg-type]
            transforms={"__time__": self._transforms["__time__"]},
        )
        sample_torch = convert_sample(sample, self._dtype)
        sample_torch = EncoderDecoderDataset._apply_masks(sample_torch)  # noqa: SLF001

        if "__time__" in sample["encoder_input"].columns:
            sample_torch["encoder_input"][0, 0] = 0.0

        sample_torch["encoder_lengths"] = torch.ones(
            (1,), dtype=self._dtype, device=sample_torch["encoder_input"].device
        )
        sample_torch["decoder_lengths"] = torch.ones(
            (1,), dtype=self._dtype, device=sample_torch["encoder_input"].device
        )

        return sample_torch

    def __iter__(self) -> typing.Generator[EncoderDecoderSample, None, None]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(self._sample_generator)


def extract_covariates_from_df(
    data: pd.DataFrame,
    columns: list[str] | str,
) -> pd.DataFrame:
    columns = columns if isinstance(columns, list) else [columns]
    return data[columns]


T = typing.TypeVar("T", np.ndarray, pd.Series, pd.DataFrame)


def keep_only_context(
    data: T,
    context_length: int,
) -> T:
    if len(data) < context_length:
        msg = "The data is shorter than the context length."
        raise ValueError(msg)
    return data[-context_length:]


U = typing.TypeVar("U", np.ndarray, pd.DataFrame)


def _apply_transforms(
    data: U,
    transforms: dict[str, BaseTransform] | BaseTransform | None = None,
    dependency: npt.NDArray[np.float64] | None = None,
) -> U:
    """
    Applies the provided transforms to the data. If the data is a 2D array,
    the transforms must be a dictionary with the column names as keys and
    the transforms as values. If the data is a 1D array or a pandas Series,
    the transforms must be a single transform.

    Parameters
    ----------
    data : np.ndarray | pd.DataFrame
        The data to be transformed.
    transforms : dict[str, BaseTransform] | BaseTransform
        The transforms to be applied to the data. If the data is a 2D array,
        the transforms must be a dictionary with the column names as keys and
        the transforms as values. If the data is a 1D array or a pandas Series,
        the transforms must be a single transform.

    Returns
    -------
    np.ndarray | pd.DataFrame
        The transformed data.
    """
    if transforms is None or (isinstance(transforms, dict) and len(transforms) == 0):
        return data
    assert transforms is not None  # mypy

    if isinstance(data, np.ndarray):  # only target will be np.ndarray
        if not isinstance(transforms, BaseTransform) and (
            isinstance(transforms, dict) and len(transforms) > 1
        ):
            msg = "More than one transform was provided for a single column."
            raise ValueError(msg)
        if isinstance(transforms, dict):
            transforms = next(iter(transforms.values()))
        assert isinstance(transforms, BaseTransform)

        if (
            dependency is not None
            and transforms.transform_type == BaseTransform.TransformType.XY
        ):
            return transforms.transform(dependency, data)
        return transforms.transform(data).numpy()

    if isinstance(transforms, BaseTransform) and len(data.columns) > 1:
        msg = "A single transform was provided for multiple columns."
        raise ValueError(msg)

    df = data.copy()
    for col, transform in transforms.items():
        df[col] = transform.transform(data[col].to_numpy()).numpy()

    return df
