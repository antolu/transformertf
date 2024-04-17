from __future__ import annotations

import functools
import typing

import torch


def chain_schedulers(
    milestones: list[int],
    *schedulers: typing.Callable[
        [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
    ],
) -> typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]:
    schedulers_l = list(schedulers)
    if len(schedulers_l) == 1:
        return schedulers_l[0]

    if len(milestones) != len(schedulers) - 1:
        msg = "Number of milestones must be one less than the number of schedulers"
        raise ValueError(msg)

    def scheduler(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            milestones=milestones,
            schedulers=[s(optimizer) for s in schedulers_l],
        )

    return functools.partial(scheduler)
