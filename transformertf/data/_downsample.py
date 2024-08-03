from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import scipy.ndimage
import torch

from .transform._utils import _as_numpy  # type: ignore[import]

__all__ = ["DOWNSAMPLE_METHODS", "downsample"]


MIN_DOWNSAMPLE_FACTOR = 1
DOWNSAMPLE_METHODS: typing.TypeAlias = typing.Literal["interval", "median", "convolve"]
ArrayLike: typing.TypeAlias = np.ndarray | torch.Tensor | pd.Series


@typing.overload
def downsample(  # type: ignore[overload-overlap]
    value: pd.DataFrame,
    downsample: int,
    method: DOWNSAMPLE_METHODS = "interval",
) -> pd.DataFrame: ...


@typing.overload
def downsample(
    value: ArrayLike,
    downsample: int,
    method: DOWNSAMPLE_METHODS = "interval",
) -> np.ndarray: ...


def downsample(
    value: ArrayLike | pd.DataFrame,
    downsample: int,
    method: DOWNSAMPLE_METHODS = "interval",
) -> np.ndarray | pd.DataFrame:
    """
    Downsamples a container of time series data.

    Parameters
    ----------
    value
    method

    Returns
    -------

    """
    if isinstance(value, np.ndarray | torch.Tensor | pd.Series):
        return downsample_array(value, downsample, method=method)
    if isinstance(value, pd.DataFrame):
        if method == "interval":
            return value.iloc[::downsample].reset_index(drop=True)

        df = value.iloc[::downsample].reset_index(drop=True)
        for col in df.columns:
            if (
                pd.api.types.is_datetime64_any_dtype(df[col].dtype)
                or pd.api.types.is_timedelta64_dtype(df[col].dtype)
                or pd.api.types.is_string_dtype(df[col].dtype)
            ):
                continue

            df[col] = downsample_array(value[col], downsample, method=method)
        return df

    msg = f"Unsupported type: {type(value)}"
    raise TypeError(msg)


def downsample_array(
    v: np.ndarray | torch.Tensor | pd.Series,
    factor: int,
    method: DOWNSAMPLE_METHODS = "interval",
) -> np.ndarray:
    """
    Downsamples a 1D array by taking the mean of each block.

    Parameters
    ----------
    v : np.ndarray | torch.Tensor | pd.Series
    factor : int
    method : {'average', 'convolve'}

    Returns
    -------

    """
    v = _as_numpy(v)

    if method == "interval":
        return v[::factor]
    if method == "median":
        return downsample_median(v, factor)
    if method == "convolve":
        return downsample_convolve(v, factor)

    msg = f"Unknown downsampling method: {method}"
    raise ValueError(msg)


def downsample_convolve(
    v: np.ndarray | torch.Tensor | pd.Series, factor: int
) -> np.ndarray:
    """
    Downsamples a 1D array by convolving with a boxcar filter.

    Parameters
    ----------
    v
    factor

    Returns
    -------

    """
    v = _as_numpy(v)
    if factor <= MIN_DOWNSAMPLE_FACTOR:
        return v

    box = np.ones(factor) / factor
    return np.convolve(v, box, mode="valid")[::factor]


def downsample_median(
    v: np.ndarray | torch.Tensor | pd.Series, factor: int
) -> np.ndarray:
    """
    Downsamples a 1D array by taking the mean of each block.

    Parameters
    ----------
    v
    factor

    Returns
    -------

    """
    v = _as_numpy(v)
    if factor <= MIN_DOWNSAMPLE_FACTOR:
        return v

    return scipy.ndimage.median_filter(v, size=factor // 2)[::factor]
