from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

DATA_SOURCE: typing.TypeAlias = pd.Series | np.ndarray | torch.Tensor

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int": torch.int64,
    "long": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}
VALID_DTYPES = typing.Literal[
    "float32",
    "float64",
    "float",
    "double",
    "float16",
    "int32",
    "int64",
    "int",
    "long",
    "int16",
    "int8",
    "uint8",
]


def get_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, str):
        return DTYPE_MAP[dtype]
    return dtype


def to_torch(
    o: pd.Series | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    if isinstance(o, pd.Series):
        return torch.from_numpy(o.to_numpy())
    if isinstance(o, np.ndarray):
        return torch.from_numpy(o)
    if isinstance(o, torch.Tensor):
        return o
    msg = f"Unsupported type {type(o)} for data"
    raise TypeError(msg)


def convert_data(
    data: DATA_SOURCE | list[DATA_SOURCE],
    dtype: torch.dtype | VALID_DTYPES = "float32",
) -> list[torch.Tensor]:
    source = data if isinstance(data, list) else [data]

    dtype = DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    return [to_torch(o).to(dtype) for o in source]
