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
    """
    TypedDict for Lightning learning rate scheduler configuration.

    This dictionary structure is used by PyTorch Lightning to configure
    learning rate schedulers with monitoring and scheduling options.

    Attributes
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
        The configured learning rate scheduler instance.
    monitor : str
        The metric name to monitor for scheduler decisions (e.g., "val_loss").
    interval : {"epoch", "step"}
        The frequency at which to update the learning rate.

    Notes
    -----
    This TypedDict is returned by `configure_lr_scheduler` when a monitor
    metric is specified, making it compatible with Lightning's scheduler
    configuration requirements.

    Examples
    --------
    >>> scheduler_config: LrSchedulerDict = {
    ...     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
    ...     "monitor": "val_loss",
    ...     "interval": "epoch"
    ... }
    """

    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    monitor: str
    interval: typing.Literal["epoch", "step"]


class OptimizerDict(typing.TypedDict):
    """
    TypedDict for Lightning optimizer configuration with optional scheduler.

    This dictionary structure is used by PyTorch Lightning to configure
    optimizers with optional learning rate schedulers in the `configure_optimizers`
    method of Lightning modules.

    Attributes
    ----------
    optimizer : torch.optim.Optimizer
        The configured optimizer instance.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | LrSchedulerDict | None
        The optional learning rate scheduler configuration. Can be:
        - A scheduler instance for simple configurations
        - A LrSchedulerDict for monitored schedulers
        - None if no scheduler is used

    Notes
    -----
    This TypedDict provides a complete optimizer configuration that can be
    returned directly from Lightning's `configure_optimizers` method.

    Examples
    --------
    >>> optimizer_config: OptimizerDict = {
    ...     "optimizer": torch.optim.Adam(model.parameters()),
    ...     "lr_scheduler": {
    ...         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
    ...         "monitor": "val_loss",
    ...         "interval": "epoch"
    ...     }
    ... }
    """

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
    Configure and return an optimizer factory function for Lightning training workflows.

    This utility function provides a unified interface for configuring common PyTorch optimizers
    with consistent parameter handling and validation. It supports both string-based optimizer
    selection and custom optimizer functions, making it easy to integrate with Lightning modules.

    Parameters
    ----------
    optimizer : str | functools.partial | typing.Callable[[typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
        The optimizer specification. Can be one of:

        - "adam": Adam optimizer with weight decay support
        - "adamw": AdamW optimizer with built-in weight decay
        - "sgd": Stochastic Gradient Descent with momentum and weight decay
        - "ranger": Ranger optimizer (from pytorch-optimizer)
        - functools.partial: Pre-configured optimizer factory function
        - Callable: Custom optimizer factory function
    lr : float | typing.Literal["auto"] | None, optional
        The learning rate for the optimizer. If "auto", defaults to 1e-3.
        Must be specified when using string-based optimizer selection.
        Ignored when optimizer is a callable.
    weight_decay : float | None, optional
        L2 penalty coefficient for regularization. Defaults to 0.0 if not specified.
        Supported by "adam", "adamw", "sgd", and "ranger" optimizers.
        Not used when optimizer is a callable.
    momentum : float | None, optional
        Momentum factor for SGD optimizer. Defaults to 0.0 if not specified.
        Only used with "sgd" optimizer. Ignored for other optimizers.
    **optimizer_kwargs : typing.Any
        Additional keyword arguments to pass to the optimizer constructor.
        These override any default parameters for the specified optimizer.

    Returns
    -------
    typing.Callable[[typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
        A factory function that takes model parameters and returns a configured optimizer.
        This function is designed to be used with Lightning's `configure_optimizers` method.

    Raises
    ------
    ValueError
        If an unknown optimizer string is specified or if learning rate is not provided
        when using string-based optimizer selection.

    Notes
    -----
    This function is commonly used in Lightning modules to configure optimizers:

    - Adam and AdamW are recommended for most deep learning tasks
    - SGD with momentum is useful for fine-tuning and specific architectures
    - Ranger combines benefits of Adam and gradient centralization
    - Custom optimizers can be provided as callables for advanced use cases

    The returned function follows the Lightning convention and can be directly
    used in the `configure_optimizers` method of Lightning modules.

    Examples
    --------
    Basic optimizer configuration:

    >>> import torch
    >>> from transformertf.utils import configure_optimizers
    >>>
    >>> model = torch.nn.Linear(10, 10)
    >>> optimizer_fn = configure_optimizers("adam", lr=1e-3, weight_decay=1e-4)
    >>> optimizer = optimizer_fn(model.parameters())
    >>> print(type(optimizer))
    <class 'torch.optim.adam.Adam'>

    Using with Lightning module:

    >>> class MyModel(LightningModule):
    ...     def configure_optimizers(self):
    ...         return configure_optimizers(
    ...             "adamw", lr=2e-4, weight_decay=1e-2
    ...         )(self.parameters())

    Custom optimizer with additional arguments:

    >>> optimizer_fn = configure_optimizers(
    ...     "adam", lr=1e-3, betas=(0.9, 0.999), eps=1e-8
    ... )
    >>> optimizer = optimizer_fn(model.parameters())

    Using a pre-configured optimizer:

    >>> import functools
    >>> custom_optimizer = functools.partial(
    ...     torch.optim.RMSprop, lr=1e-3, alpha=0.99
    ... )
    >>> optimizer_fn = configure_optimizers(custom_optimizer)
    >>> optimizer = optimizer_fn(model.parameters())

    See Also
    --------
    configure_lr_scheduler : Configure learning rate schedulers
    transformertf.models.LightningModuleBase : Base class for Lightning modules
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
    Configure and return a learning rate scheduler for Lightning training workflows.

    This utility function provides a unified interface for configuring learning rate schedulers
    with proper Lightning integration. It supports both built-in scheduler patterns and custom
    scheduler configurations, with automatic handling of monitoring and scheduling intervals.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer instance to attach the scheduler to. This should be the optimizer
        returned by the `configure_optimizers` function.
    lr_scheduler : str | type[torch.optim.lr_scheduler.LRScheduler] | functools.partial | typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        The learning rate scheduler specification. Can be one of:

        - "plateau": ReduceLROnPlateau scheduler that reduces LR when metric plateaus
        - "constant_then_cosine": Sequential scheduler with constant LR followed by cosine annealing
        - type: A scheduler class to instantiate with the optimizer
        - functools.partial: Pre-configured scheduler factory function
        - Callable: Custom scheduler factory function
    monitor : str | None, optional
        The metric name to monitor for scheduler decisions. When provided, returns a
        LrSchedulerDict suitable for Lightning's `configure_optimizers` method.
        When None, returns the raw scheduler object. Common values include
        "val_loss", "train_loss", "val_mae", etc.
    scheduler_interval : {"step", "epoch"}, optional
        The frequency at which to update the learning rate:

        - "epoch": Update after each training epoch (default)
        - "step": Update after each training step/batch
    max_epochs : int | None, optional
        Total number of training epochs. Required when using "constant_then_cosine"
        scheduler to properly configure the transition point and cosine annealing duration.
    reduce_on_plateau_patience : int | None, optional
        Number of epochs with no improvement after which learning rate will be reduced.
        Used with "plateau" scheduler. Defaults to 0 if not specified.
    **lr_scheduler_kwargs : typing.Any
        Additional keyword arguments to pass to the scheduler constructor.
        These override any default parameters for the specified scheduler.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler | LrSchedulerDict
        The configured learning rate scheduler. Return type depends on `monitor` parameter:

        - When `monitor` is None: Returns the raw scheduler object
        - When `monitor` is provided: Returns a LrSchedulerDict with keys:
          * "scheduler": The configured scheduler object
          * "monitor": The metric name to monitor
          * "interval": The scheduling interval ("step" or "epoch")

    Raises
    ------
    ValueError
        If an unknown scheduler string is provided, or if required parameters
        are missing (e.g., max_epochs for "constant_then_cosine").
    NotImplementedError
        If the scheduler type is not yet implemented.

    Notes
    -----
    Learning rate scheduling strategies:

    - **Plateau scheduling**: Reduces LR when validation metrics plateau,
      useful for adaptive learning rate adjustment
    - **Constant then cosine**: Maintains constant LR for initial training,
      then applies cosine annealing for fine-tuning
    - **Custom schedulers**: Advanced users can provide custom scheduler functions
      for specialized training regimes

    The function integrates seamlessly with Lightning's training workflow and
    automatically handles the different scheduler interfaces and monitoring requirements.

    Examples
    --------
    Basic plateau scheduler for validation loss monitoring:

    >>> import torch
    >>> from transformertf.utils import configure_optimizers, configure_lr_scheduler
    >>>
    >>> model = torch.nn.Linear(10, 10)
    >>> optimizer = configure_optimizers("adam", lr=1e-3)(model.parameters())
    >>> lr_scheduler = configure_lr_scheduler(
    ...     optimizer, "plateau", monitor="val_loss", reduce_on_plateau_patience=5
    ... )
    >>> print(lr_scheduler["scheduler"])
    ReduceLROnPlateau(...)

    Constant then cosine scheduler for long training:

    >>> lr_scheduler = configure_lr_scheduler(
    ...     optimizer, "constant_then_cosine", max_epochs=100
    ... )
    >>> # 75% constant LR, then 25% cosine annealing

    Step-based scheduling for fine-grained control:

    >>> lr_scheduler = configure_lr_scheduler(
    ...     optimizer, "plateau",
    ...     monitor="train_loss",
    ...     scheduler_interval="step"
    ... )

    Custom scheduler with additional parameters:

    >>> import functools
    >>> custom_scheduler = functools.partial(
    ...     torch.optim.lr_scheduler.CosineAnnealingLR, T_max=50
    ... )
    >>> lr_scheduler = configure_lr_scheduler(
    ...     optimizer, custom_scheduler, monitor="val_loss"
    ... )

    Using in Lightning module:

    >>> class MyModel(LightningModule):
    ...     def configure_optimizers(self):
    ...         optimizer = configure_optimizers("adam", lr=1e-3)(self.parameters())
    ...         scheduler = configure_lr_scheduler(
    ...             optimizer, "plateau", monitor="val_loss"
    ...         )
    ...         return {"optimizer": optimizer, "lr_scheduler": scheduler}

    See Also
    --------
    configure_optimizers : Configure optimizers for Lightning training
    chain_schedulers : Chain multiple schedulers with milestones
    transformertf.models.LightningModuleBase : Base class for Lightning modules
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
