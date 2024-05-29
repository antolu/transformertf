from __future__ import annotations

import typing

import torch

__all__ = [
    "BWOutput1",
    "BWOutput2",
    "BWOutput3",
    "BWOutput12",
    "BWOutput123",
    "BWState1",
    "BWState2",
    "BWState3",
    "LSTMState",
]


LSTMState: typing.TypeAlias = tuple[torch.Tensor, torch.Tensor]


class BWOutput1(typing.TypedDict):
    z: torch.Tensor
    b: typing.NotRequired[torch.Tensor]


class BWOutput2(typing.TypedDict):
    dz_dt: torch.Tensor
    g: torch.Tensor
    g_gamma_x: torch.Tensor


class BWOutput3(typing.TypedDict):
    dr_dt: torch.Tensor


class BWState1(typing.TypedDict):
    hx: LSTMState


class BWState2(BWState1):
    hx2: LSTMState


class BWState3(BWState2):
    hx3: LSTMState


class BWOutput12(BWOutput1, BWOutput2):  # type: ignore[misc]
    pass


class BWOutput123(BWOutput12, BWOutput3):  # type: ignore[misc]
    pass


class BWLoss1(typing.TypedDict):
    loss1: torch.Tensor
    loss2: torch.Tensor


class BWLoss2(BWLoss1):  # type: ignore[misc]
    loss3: torch.Tensor
    loss4: torch.Tensor


class BWLoss3(BWLoss2):  # type: ignore[misc]
    loss5: torch.Tensor


HiddenStateNone: typing.TypeAlias = (
    list[BWState1 | None] | list[BWState2 | None] | list[BWState3 | None]
)
BWLSTMOutput: typing.TypeAlias = BWOutput1 | BWOutput12 | BWOutput123
BWLSTMStates: typing.TypeAlias = BWState1 | BWState2 | BWState3
