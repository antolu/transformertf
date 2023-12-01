from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

__all__ = ["downsample"]


if typing.TYPE_CHECKING:
    T = typing.TypeVar("T", pd.Series, np.ndarray, torch.Tensor, pd.DataFrame)


def downsample(
    value: T, downsample: int, method: typing.Literal["interval"] = "interval"
) -> T:
    """
    Downsamples a container of time series data.

    Parameters
    ----------
    value
    method

    Returns
    -------

    """
    if isinstance(value, (np.ndarray, torch.Tensor, pd.Series)):
        return value[::downsample]
    elif isinstance(value, pd.DataFrame):
        return value.iloc[::downsample].reset_index()
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
