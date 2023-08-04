from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import torch


__all__ = [
    "PhyLSTM1Output",
    "PhyLSTM2Output",
    "PhyLSTM3Output",
    "PhyLSTM1States",
    "PhyLSTM2States",
    "PhyLSTM3States",
]


class PhyLSTM1Output(TypedDict):
    z: torch.Tensor
    z_raw: torch.Tensor


class PhyLSTM2Output(PhyLSTM1Output):
    dz_dt: torch.Tensor
    g: torch.Tensor
    g_gamma_x: torch.Tensor


class PhyLSTM3Output(PhyLSTM2Output):
    dr_dt: torch.Tensor


class PhyLSTM1States(TypedDict):
    lstm1: tuple[torch.Tensor, ...]


class PhyLSTM2States(PhyLSTM1States):
    lstm2: tuple[torch.Tensor, ...]


class PhyLSTM3States(PhyLSTM2States):
    lstm3: tuple[torch.Tensor, ...]
