from __future__ import annotations

import typing

import torch

MODEL_INPUT: typing.TypeAlias = torch.Tensor | dict[str, torch.Tensor]
MODEL_OUTPUT: typing.TypeAlias = torch.Tensor | dict[str, torch.Tensor]
MODEL_STATES: typing.TypeAlias = torch.Tensor | dict[str, torch.Tensor]

STEP_OUTPUT: typing.TypeAlias = MODEL_OUTPUT | dict[str, MODEL_OUTPUT]
EPOCH_OUTPUT: typing.TypeAlias = list[STEP_OUTPUT]

OPT_CALL_TYPE: typing.TypeAlias = typing.Callable[
    [tuple[typing.Any, ...]], torch.optim.Optimizer
]
LR_CALL_TYPE: typing.TypeAlias = (
    torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
)
