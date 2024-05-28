from __future__ import annotations

from typing import TypedDict

import torch

__all__ = [
    "BoucWenOutput1",
    "BoucWenOutput2",
    "BoucWenOutput3",
    "BoucWenStates1",
    "BoucWenStates2",
    "BoucWenStates3",
]


class BoucWenOutput1(TypedDict):
    z: torch.Tensor
    b: torch.Tensor


class BoucWenOutput2(BoucWenOutput1):
    dz_dt: torch.Tensor
    g: torch.Tensor
    g_gamma_x: torch.Tensor


class BoucWenOutput3(BoucWenOutput2):
    dr_dt: torch.Tensor


class BoucWenStates1(TypedDict):
    lstm1: tuple[torch.Tensor, ...]


class BoucWenStates2(BoucWenStates1):
    lstm2: tuple[torch.Tensor, ...]


class BoucWenStates3(BoucWenStates2):
    lstm3: tuple[torch.Tensor, ...]
