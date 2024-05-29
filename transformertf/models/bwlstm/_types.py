from __future__ import annotations

import typing

import torch

__all__ = [
    "BoucWenOutput1",
    "BoucWenOutput2",
    "BoucWenOutput3",
    "BoucWenStates1",
    "BoucWenStates2",
    "BoucWenStates3",
    "LSTMState",
]


LSTMState: typing.TypeAlias = tuple[torch.Tensor, torch.Tensor]


class BoucWenOutput1(typing.TypedDict):
    z: torch.Tensor
    b: typing.NotRequired[torch.Tensor]


class BoucWenOutput2(typing.TypedDict):
    dz_dt: torch.Tensor
    g: torch.Tensor
    g_gamma_x: torch.Tensor


class BoucWenOutput3(typing.TypedDict):
    dr_dt: torch.Tensor


class BoucWenStates1(typing.TypedDict):
    hx: LSTMState


class BoucWenStates2(BoucWenStates1):
    hx2: LSTMState


class BoucWenStates3(BoucWenStates2):
    hx3: LSTMState


class BoucWenOutput12(BoucWenOutput1, BoucWenOutput2):  # type: ignore[misc]
    pass


class BoucWenOutput123(BoucWenOutput12, BoucWenOutput3):  # type: ignore[misc]
    pass


class BWLoss1(typing.TypedDict):
    loss1: torch.Tensor
    loss2: torch.Tensor


class BWLoss2(BWLoss1):  # type: ignore[misc]
    loss3: torch.Tensor
    loss4: torch.Tensor


class BWLoss3(BWLoss2):  # type: ignore[misc]
    loss5: torch.Tensor
