from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _view_as_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim < y.ndim:
        return x.view(*(1,) * (y.ndim - x.ndim), *x.size())
    return x


def _as_torch(
    x: list[float] | pd.Series | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    if isinstance(x, list):
        x = torch.tensor(x)
    elif isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.numpy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
