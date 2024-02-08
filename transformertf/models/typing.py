import typing

import torch

MODEL_INPUT = typing.Union[torch.Tensor, dict[str, torch.Tensor]]
MODEL_OUTPUT = typing.Union[torch.Tensor, dict[str, torch.Tensor]]
MODEL_STATES = typing.Union[torch.Tensor, dict[str, torch.Tensor]]

STEP_OUTPUT = typing.Union[MODEL_OUTPUT, dict[str, MODEL_OUTPUT]]
EPOCH_OUTPUT = list[STEP_OUTPUT]

OPT_CALL_TYPE = typing.Callable[
    [tuple[typing.Any, ...]], torch.optim.Optimizer
]
LR_CALL_TYPE = typing.Callable[
    [tuple[typing.Any, ...]], torch.optim.lr_scheduler.LRScheduler
]
