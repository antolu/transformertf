"""
This module contains functions to configure the optimizer and learning rate scheduler.
"""

from __future__ import annotations

import functools
import logging
import typing

import pytorch_optimizer as py_optim
import torch

log = logging.getLogger(__name__)


class LrSchedulerDict(typing.TypedDict):
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    monitor: str
    interval: typing.Literal["epoch", "step"]


class OptimizerDict(typing.TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | LrSchedulerDict
        | None
    )


__all__ = [
    "LrSchedulerDict",
    "OptimizerDict",
    "configure_lr_scheduler",
    "configure_optimizers",
]


def configure_optimizers(
    optimizer: str
    | functools.partial
    | typing.Callable[[typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
    lr: float | typing.Literal["auto"] | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    **optimizer_kwargs: typing.Any,
) -> typing.Callable[[typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer]:
    """
    Return a function that takes an iterator of torch.nn.Parameter and returns a torch.optim.Optimizer object.

    Parameters
    ----------
    optimizer : str | functools.partial | typing.Callable[[typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
        The optimizer to use. Supported optimizers are: "adam", "adamw", "sgd", "ranger".
        If a functools.partial is passed, it is assumed that the partial is a function that takes an iterator of
        torch.nn.Parameter and returns a torch.optim.Optimizer object.
    lr : float
        The optimizer learning rate.
    weight_decay : float | None, optional
        The weight decay. Defaults to None. Only used for "adam" and "sgd" optimizers.
    momentum : float | None, optional
        The momentum. Defaults to None. Only used for "sgd" optimizer.
    **optimizer_kwargs : typing.Any
        Additional optimizer keyword arguments.

    Returns
    -------
    typing.Callable[[typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
        A function that takes an iterator of torch.nn.Parameter and returns a torch.optim.Optimizer object.

    Raises
    ------
    ValueError
        If an unknown optimizer is specified.

    Examples
    --------
    >>> import torch
    >>> from transformertf.utils import configure_optimizers
    >>>
    >>> model = torch.nn.Linear(10, 10)
    >>> optimizer = configure_optimizers("adam", lr=1e-3)(model.parameters())
    """
    # if optimizer is callable
    if callable(optimizer):
        return optimizer

    if lr is None:
        msg = "lr must be specified if optimizer is not callable."
        raise ValueError(msg)
    if lr == "auto":
        lr = 1e-3

    if optimizer == "adam":
        return functools.partial(
            torch.optim.Adam,
            lr=lr,
            weight_decay=weight_decay or 0.0,
            **optimizer_kwargs,
        )
    if optimizer == "adamw":
        return functools.partial(
            torch.optim.AdamW,
            lr=lr,
            weight_decay=weight_decay or 0.0,
            **optimizer_kwargs,
        )
    if optimizer == "sgd":
        return functools.partial(
            torch.optim.SGD,
            lr=lr,
            weight_decay=weight_decay or 0.0,
            momentum=momentum or 0.0,
            **optimizer_kwargs,
        )
    if optimizer == "ranger":
        return functools.partial(
            py_optim.Ranger,
            lr=lr,
            weight_decay=weight_decay or 0.0,
            **optimizer_kwargs,
        )

    msg = f"Unknown optimizer: {optimizer}"
    raise ValueError(msg)


def configure_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: (
        str
        | type[torch.optim.lr_scheduler.LRScheduler]
        | functools.partial
        | typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
    ),
    monitor: str | None = None,
    scheduler_interval: typing.Literal["step", "epoch"] = "epoch",
    max_epochs: int | None = None,
    reduce_on_plateau_patience: int | None = None,
    **lr_scheduler_kwargs: typing.Any,
) -> LrSchedulerDict | torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Configure and return a learning rate scheduler for the optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule the learning rate for.
    lr_scheduler : str | typing.Type[torch.optim.lr_scheduler.LRScheduler] | functools.partial | typing.Callable[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        The learning rate scheduler to configure. It can be one of the following:
        - A string: "plateau" or "constant_then_cosine".
        - A class or a partial function that returns an instance of `torch.optim.lr_scheduler.LRScheduler`.
        - A callable function that takes an optimizer as input and returns an instance of `torch.optim.lr_scheduler.LRScheduler`.
        - `None` to indicate no learning rate scheduler.
    monitor : str, optional
        The metric to monitor for the scheduler. If `None`, the function returns the configured scheduler,
        otherwise it returns a dictionary containing the configured scheduler and its parameters, configure for
        use with `pytorch_lightning.Trainer`. Defaults to `None`.
    scheduler_interval : typing.Literal["step", "epoch"], optional
        The interval at which the scheduler should be updated. Defaults to "epoch".
    max_epochs : int | None, optional
        The maximum number of epochs. Required when using the "constant_then_cosine" lr_scheduler. Defaults to `None`.
    reduce_on_plateau_patience : int | None, optional
        The number of epochs with no improvement after which learning rate will be reduced. Required when using the "plateau" lr_scheduler. Defaults to `None`.

    Returns
    -------
     torch.optim.lr_scheduler.LRScheduler | LR_SCHEDULER_DICT
        The configure learning rate scheduler, or a dictionary containing the
        configured learning rate scheduler and its parameters. The return
        type is a dictionary when `monitor` is not `None`.

    Raises
    ------
    ValueError
        If an unknown `lr_scheduler` is provided.
        If `max_epochs` is not specified when using the "constant_then_cosine" lr_scheduler.
    NotImplementedError
        If the learning rate schedulers are not implemented yet.

    Examples
    --------
    >>> import torch
    >>> from transformertf.utils import configure_optimizers, configure_lr_scheduler
    >>>
    >>> model = torch.nn.Linear(10, 10)
    >>> optimizer = configure_optimizers("adam", lr=1e-3)(model.parameters())
    >>> lr_scheduler = configure_lr_scheduler(optimizer, "plateau")
    >>> scheduler = lr_scheduler["lr_scheduler"]
    """
    if isinstance(lr_scheduler, str):
        if lr_scheduler == "plateau":
            if reduce_on_plateau_patience is None:
                log.warning(
                    "reduce_on_plateau_patience is not specified, using 0 as default."
                )
                reduce_on_plateau_patience = 0
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=0.1,
                patience=reduce_on_plateau_patience,
            )
        elif lr_scheduler == "constant_then_cosine":
            if max_epochs is None:
                msg = (
                    "max_epochs must be specified when using the "
                    "constant_then_cosine lr_scheduler."
                )
                raise ValueError(msg)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [  # type: ignore
                    torch.optim.lr_scheduler.ConstantLR(
                        factor=1.0,
                        optimizer=optimizer,
                        total_iters=int(0.75 * max_epochs),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=int(0.25 * max_epochs),
                    ),
                ],
                milestones=[int(0.75 * max_epochs)],
            )
        else:
            msg = f"Unknown lr_scheduler: {lr_scheduler}"
            raise ValueError(msg)
    elif isinstance(lr_scheduler, functools.partial) or callable(lr_scheduler):
        scheduler = lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)  # type: ignore
    else:
        msg = "Learning rate schedulers are not implemented yet."
        raise NotImplementedError(msg)

    if monitor is None:
        return scheduler

    return typing.cast(
        LrSchedulerDict,
        {
            "scheduler": scheduler,
            "monitor": monitor,
            "interval": scheduler_interval,
        },
    )
