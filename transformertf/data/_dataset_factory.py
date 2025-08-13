"""
Dataset Factory for creating datasets with consistent APIs.

This module provides factory methods for creating different types of datasets
with explicit parameters while reducing code duplication across data modules.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ._covariates import (
    KNOWN_COV_CONT_PREFIX,
    PAST_KNOWN_COV_PREFIX,
    TARGET_PREFIX,
    TIME_PREFIX,
)
from ._dtype import VALID_DTYPES
from .dataset import (
    EncoderDecoderDataset,
    TimeSeriesDataset,
)
from .transform import BaseTransform

__all__ = ["DatasetFactory"]


class DatasetFactory:
    """
    Factory for creating datasets with consistent APIs.

    This factory provides static methods for creating different types of
    datasets while maintaining explicit parameter interfaces required by
    Lightning data modules.
    """

    @staticmethod
    def create_timeseries_dataset(
        data: pd.DataFrame | list[pd.DataFrame],
        *,
        seq_len: int,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        predict: bool = False,
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
    ) -> TimeSeriesDataset:
        """
        Create TimeSeriesDataset with explicit parameters.

        Parameters
        ----------
        data : pd.DataFrame | list[pd.DataFrame]
            Input data for the dataset.
        seq_len : int
            Length of each sequence sample.
        min_seq_len : int | None, optional
            Minimum sequence length when using randomized lengths.
        randomize_seq_len : bool, optional
            Whether to randomize sequence lengths during training.
        stride : int, optional
            Step size for sliding window. Default is 1.
        predict : bool, optional
            Whether dataset is used for prediction. Default is False.
        transforms : dict[str, BaseTransform] | None, optional
            Transforms to apply to data.
        dtype : VALID_DTYPES, optional
            Data type for tensors. Default is "float32".

        Returns
        -------
        TimeSeriesDataset
            Configured time series dataset.
        """
        # Extract input and target data
        if isinstance(data, list):
            input_data = [
                df.drop(
                    columns=[
                        col for col in df.columns if col.startswith(TARGET_PREFIX)
                    ],
                )
                for df in data
            ]
            target_data = [
                df[[col for col in df.columns if col.startswith(TARGET_PREFIX)]]
                for df in data
                if any(col.startswith(TARGET_PREFIX) for col in df.columns)
            ]
            if not target_data:
                target_data = None
        else:
            input_cols = [
                col for col in data.columns if not col.startswith(TARGET_PREFIX)
            ]
            target_cols = [col for col in data.columns if col.startswith(TARGET_PREFIX)]

            input_data = data[input_cols] if input_cols else data
            target_data = data[target_cols] if target_cols else None

        return TimeSeriesDataset(
            input_data=input_data,
            target_data=target_data,
            seq_len=seq_len,
            min_seq_len=min_seq_len,
            randomize_seq_len=randomize_seq_len,
            stride=stride,
            predict=predict,
            transforms=transforms,
            dtype=dtype,
        )

    @staticmethod
    def create_encoder_decoder_dataset(
        data: pd.DataFrame | list[pd.DataFrame],
        *,
        ctx_seq_len: int,
        tgt_seq_len: int,
        min_ctx_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        predict: bool = False,
        transforms: dict[str, BaseTransform] | None = None,
        dtype: VALID_DTYPES = "float32",
        time_format: Literal["relative", "absolute"] = "relative",
        noise_std: float = 0.0,
        add_target_to_past: bool = True,
        masked_encoder_features: list[str] | None = None,
    ) -> EncoderDecoderDataset:
        """
        Create EncoderDecoderDataset with explicit parameters.

        Parameters
        ----------
        data : pd.DataFrame | list[pd.DataFrame]
            Input data for the dataset.
        ctx_seq_len : int
            Context (encoder) sequence length.
        tgt_seq_len : int
            Target (decoder) sequence length.
        min_ctx_seq_len : int | None, optional
            Minimum context sequence length for randomization.
        min_tgt_seq_len : int | None, optional
            Minimum target sequence length for randomization.
        randomize_seq_len : bool, optional
            Whether to randomize sequence lengths.
        stride : int, optional
            Step size for sliding window. Default is 1.
        predict : bool, optional
            Whether dataset is used for prediction. Default is False.
        transforms : dict[str, BaseTransform] | None, optional
            Transforms to apply to data.
        dtype : VALID_DTYPES, optional
            Data type for tensors. Default is "float32".
        time_format : {"relative", "absolute"}, optional
            Time format for temporal features. Default is "relative".
        noise_std : float, optional
            Standard deviation for noise injection. Default is 0.0.
        add_target_to_past : bool, optional
            Whether to add target to past context. Default is True.
        masked_encoder_features : list[str] | None, optional
            List of encoder feature names to mask (zero out). Default is None.

        Returns
        -------
        EncoderDecoderDataset
            Configured encoder-decoder dataset.
        """
        # Extract different data types
        if isinstance(data, list):
            input_data = [_extract_columns(df, KNOWN_COV_CONT_PREFIX) for df in data]
            known_past_data = [
                _extract_columns(df, PAST_KNOWN_COV_PREFIX) for df in data
            ]
            target_data = [_extract_columns(df, TARGET_PREFIX) for df in data]
            time_data = [_extract_columns(df, TIME_PREFIX) for df in data]
        else:
            input_data = _extract_columns(data, KNOWN_COV_CONT_PREFIX)
            known_past_data = _extract_columns(data, PAST_KNOWN_COV_PREFIX)
            target_data = _extract_columns(data, TARGET_PREFIX)
            time_data = _extract_columns(data, TIME_PREFIX)

        # Handle None values for optional data
        if (
            isinstance(known_past_data, list)
            and all(df.empty for df in known_past_data)
        ) or (isinstance(known_past_data, pd.DataFrame) and known_past_data.empty):
            known_past_data = None

        if (isinstance(time_data, list) and all(df.empty for df in time_data)) or (
            isinstance(time_data, pd.DataFrame) and time_data.empty
        ):
            time_data = None

        return EncoderDecoderDataset(
            input_data=input_data,
            target_data=target_data,
            known_past_data=known_past_data,
            time_data=time_data,
            ctx_seq_len=ctx_seq_len,
            tgt_seq_len=tgt_seq_len,
            min_ctxt_seq_len=min_ctx_seq_len,
            min_tgt_seq_len=min_tgt_seq_len,
            randomize_seq_len=randomize_seq_len,
            stride=stride,
            predict=predict,
            transforms=transforms,
            dtype=dtype,
            time_format=time_format,
            noise_std=noise_std,
            add_target_to_past=add_target_to_past,
            masked_encoder_features=masked_encoder_features,
        )


def _extract_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Extract columns with given prefix from dataframe."""
    matching_cols = [col for col in df.columns if col.startswith(prefix)]
    return df[matching_cols] if matching_cols else pd.DataFrame()
